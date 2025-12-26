import os
import logging
import re
import time
import random
import collections
import threading
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Literal, Any

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AzureOpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    AzureOpenAI = None


class _GroqRateLimiter:
    """Process-wide rate limiter for Groq requests.

    Shared across all `GroqTranslator` instances; safe for multi-threaded use.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_times = collections.deque()  # monotonic timestamps
        self._last_request_time: float = 0.0
        self._configured = False

        self._rpm: int = 0
        self._window_seconds: float = 60.0
        self._min_delay_seconds: float = 0.0
        self._safety_seconds: float = 0.5
        self._effective_rpm: int = 0

    def configure(
        self,
        *,
        rpm: int,
        window_seconds: float,
        min_delay_seconds: float,
        safety_seconds: float,
        effective_rpm: int,
    ) -> None:
        with self._lock:
            self._rpm = max(0, int(rpm))
            self._window_seconds = max(1.0, float(window_seconds))
            self._min_delay_seconds = max(0.0, float(min_delay_seconds))
            self._safety_seconds = max(0.0, float(safety_seconds))
            self._effective_rpm = int(effective_rpm)
            self._configured = True

    def _allowed_rpm(self) -> int:
        if self._rpm <= 0:
            return 0
        if self._effective_rpm > 0:
            return max(1, min(self._rpm, self._effective_rpm))
        return max(1, self._rpm - 1)

    def acquire(self) -> None:
        if not self._configured:
            return

        while True:
            now = time.monotonic()
            sleep_for = 0.0

            with self._lock:
                if self._min_delay_seconds > 0:
                    since_last = now - self._last_request_time
                    if since_last < self._min_delay_seconds:
                        sleep_for = max(
                            sleep_for,
                            self._min_delay_seconds - since_last,
                        )

                allowed = self._allowed_rpm()
                if allowed > 0:
                    window = self._window_seconds
                    while (
                        self._request_times
                        and (now - self._request_times[0]) >= window
                    ):
                        self._request_times.popleft()

                    if len(self._request_times) >= allowed:
                        oldest = self._request_times[0]
                        until_ok = window - (now - oldest)
                        sleep_for = max(
                            sleep_for,
                            until_ok + self._safety_seconds,
                        )

                if sleep_for <= 0:
                    self._last_request_time = now
                    if allowed > 0:
                        self._request_times.append(now)
                    return

            logger.warning(
                "Groq rate limit reached. Sleeping %.2fs",
                sleep_for,
            )
            time.sleep(sleep_for)


_GROQ_LIMITER = _GroqRateLimiter()


class AbstractTranslator(ABC):
    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        pass

    def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
        register_hint: Optional[str] = None,
    ) -> str:
        """Translate a segment using surrounding context.

        Default implementation falls back to `translate` for translators that
        don't support context.
        """

        _ = previous_segments, next_segments, register_hint
        return self.translate(text, source_language, target_language)


def infer_register_from_asr(
    text: str,
) -> Literal["colloquial", "formal", "neutral"]:
    """Heuristic register detector based on ASR text.

    This is intentionally simple and language-agnostic-ish: it's used only to
    nudge the LLM to keep the same register in translation.
    """

    if not text:
        return "neutral"

    t = text.strip().lower()

    colloquial_markers = [
        "gonna",
        "wanna",
        "gotta",
        "ain't",
        "kinda",
        "sorta",
        "lemme",
        "dunno",
        "yeah",
        "yep",
        "nah",
        "lol",
    ]
    formal_markers = [
        "therefore",
        "however",
        "moreover",
        "furthermore",
        "regarding",
        "sincerely",
        "respectfully",
        "please",
    ]

    colloquial_score = sum(1 for m in colloquial_markers if m in t)
    formal_score = sum(1 for m in formal_markers if m in t)

    # Punctuation and contraction hints.
    contraction_pattern = (
        r"\b(i'm|you're|we're|they're|"
        r"can't|won't|don't|doesn't|isn't|aren't)\b"
    )
    if re.search(contraction_pattern, t):
        colloquial_score += 1
    if any(ch in t for ch in ["!!!", "?!", "..", "..."]):
        colloquial_score += 1

    if formal_score > colloquial_score and formal_score >= 2:
        return "formal"
    if colloquial_score > formal_score and colloquial_score >= 2:
        return "colloquial"
    return "neutral"


def build_context_aware_translation_prompt(
    *,
    source_language: str,
    target_language: str,
    current_text: str,
    previous_segments: Sequence[str],
    next_segments: Sequence[str],
    register: Literal["colloquial", "formal", "neutral"],
) -> str:
    prev_block = (
        "\n".join(f"- {t}" for t in previous_segments)
        if previous_segments
        else "(none)"
    )
    next_block = (
        "\n".join(f"- {t}" for t in next_segments)
        if next_segments
        else "(none)"
    )

    register_instruction = {
        "colloquial": "Keep it colloquial and natural, like casual speech.",
        "formal": "Keep it formal and polite.",
        "neutral": "Keep it natural and neutral (not too formal or slangy).",
    }[register]

    return (
        "You are a professional audiovisual translator.\n"
        f"Translate ONLY the CURRENT segment from {source_language} "
        f"to {target_language}.\n"
        "Use surrounding context ONLY to resolve meaning and references.\n"
        "Do NOT translate context lines; translate only the CURRENT segment.\n"
        "Preserve meaning, punctuation, numbers, and named entities.\n"
        f"Register guidance: {register_instruction}\n\n"
        "PREVIOUS (up to 5 segments):\n"
        f"{prev_block}\n\n"
        "CURRENT (translate this):\n"
        f"{current_text}\n\n"
        "NEXT (up to 5 segments):\n"
        f"{next_block}\n\n"
        "Output only the translated CURRENT segment text."
    )


@dataclass
class TranslationQuality:
    status: str
    score: float | None = None
    summary: str | None = None
    issues: list[str] | None = None
    model: str | None = None


def _extract_first_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""

    t = (text or "").strip()
    if not t:
        raise ValueError("empty response")

    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def _clamp01(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_register(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"formal", "colloquial", "neutral", "casual"}:
        return "colloquial" if s == "casual" else s
    return "neutral"


def _openai_retry_sleep_seconds(attempt: int) -> float:
    return min(10.0, (2.0 ** attempt) * 0.8)


def _normalize_azure_endpoint(endpoint: str) -> str:
    e = (endpoint or "").strip()
    if len(e) >= 2 and e[0] == e[-1] and e[0] in {'"', "'"}:
        e = e[1:-1].strip()
    # Users often paste full REST base paths; SDK expects resource root.
    for suffix in ("/openai/v1/", "/openai/v1", "/openai/", "/openai"):
        if e.lower().endswith(suffix):
            e = e[: -len(suffix)]
            break
    return e.rstrip("/")


class OpenAITranslator(AbstractTranslator):
    """Translate text using OpenAI chat completion API.

    Requires OPENAI_API_KEY environment variable set.
    
    To use OpenAI without a paid subscription:
    1. Use free trial credits (if available in your region)
    2. Use LLM APIs with free tiers:
       - Claude (Anthropic) - has free tier
       - Ollama (local) - completely free
       - Groq - free API tier
    
    See GroqTranslator or OllamaTranslator for free alternatives.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        if OpenAI is None:
            raise RuntimeError(
                "openai package not installed; "
                "install requirements to use OpenAITranslator"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable "
                "or pass api_key parameter. "
                "For free alternatives, see GroqTranslator or "
                "OllamaTranslator."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt = (
            f"You are a helpful translator. Translate the following from "
            f"{source_language} to {target_language} while preserving "
            f"meaning, punctuation and named entities.\n\nText:\n{text}"
        )

        logger.info(
            "Sending translation request to OpenAI model %s",
            self.model,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            out = resp.choices[0].message.content.strip()
            return out
        except Exception as e:
            logger.error("OpenAI translation failed: %s", str(e))
            raise

    def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
        register_hint: Optional[str] = None,
    ) -> str:
        # If a register is explicitly requested, keep the existing prompt path.
        if register_hint in {"colloquial", "formal", "neutral"}:
            prompt = build_context_aware_translation_prompt(
                source_language=source_language,
                target_language=target_language,
                current_text=text,
                previous_segments=list(previous_segments or []),
                next_segments=list(next_segments or []),
                register=register_hint,  # type: ignore[arg-type]
            )

            logger.info(
                "Sending context-aware translation request to OpenAI model %s",
                self.model,
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return resp.choices[0].message.content.strip()

        out = self.translate_with_context_and_quality(
            text=text,
            source_language=source_language,
            target_language=target_language,
            previous_segments=previous_segments,
            next_segments=next_segments,
        )
        return (out.get("text") or "").strip()

    def translate_with_context_and_quality(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
    ) -> dict:
        """Translate + auto-localize + infer register + score quality.

        The quality score is a best-effort self-evaluation to help triage.
        """

        prev_block = (
            "\n".join(f"- {t}" for t in (previous_segments or []))
            if previous_segments
            else "(none)"
        )
        next_block = (
            "\n".join(f"- {t}" for t in (next_segments or []))
            if next_segments
            else "(none)"
        )

        system = (
            "You are a professional audiovisual translator and localizer. "
            "Translate dialogue for dubbing. Preserve meaning, "
            "named entities, "
            "and numbers. "
            "Keep it natural, timing-friendly, and consistent with the "
            "speaker's tone. "
            "Automatically choose the best register in the TARGET language "
            "(formal/colloquial/neutral) based on the CURRENT segment and "
            "surrounding context. "
            "Localize naturally for the target language (idioms/phrasing) "
            "without adding new facts. "
            "Return strict JSON only."
        )

        user = (
            f"SOURCE_LANGUAGE: {source_language}\n"
            f"TARGET_LANGUAGE: {target_language}\n\n"
            "PREVIOUS (context only, do not translate):\n"
            f"{prev_block}\n\n"
            "CURRENT (translate this only):\n"
            f"{text}\n\n"
            "NEXT (context only, do not translate):\n"
            f"{next_block}\n\n"
            "Output JSON:\n"
            "{\n"
            "  \"translation\": string,\n"
            "  \"register\": \"formal\"|\"colloquial\"|\"neutral\",\n"
            "  \"localization_notes\": string,\n"
            "  \"quality\": {\n"
            "    \"score\": number between 0 and 1,\n"
            "    \"summary\": string,\n"
            "    \"issues\": [string]\n"
            "  }\n"
            "}"
        )

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2000,
                }
                # Ask for strict JSON where supported.
                kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                obj = _extract_first_json_object(content)

                translation = str(obj.get("translation") or "").strip()
                register = _normalize_register(obj.get("register"))
                localization_notes = (
                    str(obj.get("localization_notes") or "").strip() or None
                )

                q = (
                    obj.get("quality")
                    if isinstance(obj.get("quality"), dict)
                    else {}
                )
                score = _clamp01(q.get("score"))
                summary = str(q.get("summary") or "").strip() or None
                raw_issues = q.get("issues")
                issues = (
                    [str(i).strip() for i in raw_issues if str(i).strip()]
                    if isinstance(raw_issues, list)
                    else None
                )

                tq = TranslationQuality(
                    status="ok" if translation else "failed",
                    score=score,
                    summary=summary,
                    issues=issues,
                    model=self.model,
                )

                return {
                    "text": translation,
                    "register": register,
                    "localization_notes": localization_notes,
                    "translation_quality": {
                        "status": tq.status,
                        "score": tq.score,
                        "summary": tq.summary,
                        "issues": tq.issues,
                        "model": tq.model,
                    },
                }
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    sleep_s = _openai_retry_sleep_seconds(attempt)
                    logger.warning(
                        "OpenAI rate-limited; sleeping %.2fs",
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                break

        raise RuntimeError(
            f"OpenAI translate_with_context_and_quality failed: {last_err}"
        )


class AzureOpenAITranslator(AbstractTranslator):
    """Translate text using Azure OpenAI API.

    Requires environment variables:
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    - AZURE_OPENAI_DEPLOYMENT: Your deployment name (e.g., 'gpt-4-turbo')

    Get free credits: Azure provides $200 free credit for 30 days.
    Sign up: https://azure.microsoft.com/en-us/free/
    """

    def __init__(
        self,
        deployment_name: str = None,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = "2024-08-01-preview",
    ):
        if AzureOpenAI is None:
            raise RuntimeError(
                "openai package not installed; "
                "install requirements to use AzureOpenAITranslator"
            )

        # Get from parameters or environment
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = (
            endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        self.endpoint = _normalize_azure_endpoint(self.endpoint)
        self.deployment_name = (
            deployment_name
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        )

        if not self.api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY not found. "
                "Set it as environment variable or pass api_key parameter. "
                "Sign up for free: https://azure.microsoft.com/en-us/free/"
            )
        if not self.endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not found. "
                "Set it as environment variable or pass endpoint parameter. "
                "Your endpoint looks like: "
                "https://your-resource.openai.azure.com/"
            )
        if not self.deployment_name:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT not found. "
                "Set it as environment variable or pass "
                "deployment_name parameter."
            )

        self.api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION") or api_version
        )
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt = (
            f"You are a helpful translator. Translate the following from "
            f"{source_language} to {target_language} while preserving "
            f"meaning, punctuation and named entities.\n\nText:\n{text}"
        )

        logger.info(
            "Sending translation to Azure OpenAI deployment %s",
            self.deployment_name,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            out = resp.choices[0].message.content.strip()
            return out
        except Exception as e:
            logger.error("Azure OpenAI translation failed: %s", str(e))
            raise

    def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
        register_hint: Optional[str] = None,
    ) -> str:
        # If a register is explicitly requested, keep the existing prompt path.
        if register_hint in {"colloquial", "formal", "neutral"}:
            prompt = build_context_aware_translation_prompt(
                source_language=source_language,
                target_language=target_language,
                current_text=text,
                previous_segments=list(previous_segments or []),
                next_segments=list(next_segments or []),
                register=register_hint,  # type: ignore[arg-type]
            )

            logger.info(
                "Sending context-aware translation to Azure OpenAI "
                "deployment %s",
                self.deployment_name,
            )
            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return resp.choices[0].message.content.strip()

        out = self.translate_with_context_and_quality(
            text=text,
            source_language=source_language,
            target_language=target_language,
            previous_segments=previous_segments,
            next_segments=next_segments,
        )
        return (out.get("text") or "").strip()

    def translate_with_context_and_quality(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
    ) -> dict:
        """Translate + auto-localize + infer register + score quality.

        The quality score is a best-effort self-evaluation to help triage.
        """

        prev_block = (
            "\n".join(f"- {t}" for t in (previous_segments or []))
            if previous_segments
            else "(none)"
        )
        next_block = (
            "\n".join(f"- {t}" for t in (next_segments or []))
            if next_segments
            else "(none)"
        )

        system = (
            "You are a professional audiovisual translator and localizer. "
            "Translate dialogue for dubbing. Preserve meaning, "
            "named entities, "
            "and numbers. "
            "Keep it natural, timing-friendly, and consistent with the "
            "speaker's tone. "
            "Automatically choose the best register in the TARGET language "
            "(formal/colloquial/neutral) based on the CURRENT segment and "
            "surrounding context. "
            "Localize naturally for the target language (idioms/phrasing) "
            "without adding new facts. "
            "Return strict JSON only."
        )

        user = (
            f"SOURCE_LANGUAGE: {source_language}\n"
            f"TARGET_LANGUAGE: {target_language}\n\n"
            "PREVIOUS (context only, do not translate):\n"
            f"{prev_block}\n\n"
            "CURRENT (translate this only):\n"
            f"{text}\n\n"
            "NEXT (context only, do not translate):\n"
            f"{next_block}\n\n"
            "Output JSON:\n"
            "{\n"
            "  \"translation\": string,\n"
            "  \"register\": \"formal\"|\"colloquial\"|\"neutral\",\n"
            "  \"localization_notes\": string,\n"
            "  \"quality\": {\n"
            "    \"score\": number between 0 and 1,\n"
            "    \"summary\": string,\n"
            "    \"issues\": [string]\n"
            "  }\n"
            "}"
        )

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                kwargs = {
                    "model": self.deployment_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_completion_tokens": 2000,
                }
                # Ask for strict JSON where supported.
                kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                obj = _extract_first_json_object(content)

                translation = str(obj.get("translation") or "").strip()
                register = _normalize_register(obj.get("register"))
                localization_notes = (
                    str(obj.get("localization_notes") or "").strip() or None
                )

                q = (
                    obj.get("quality")
                    if isinstance(obj.get("quality"), dict)
                    else {}
                )
                score = _clamp01(q.get("score"))
                summary = str(q.get("summary") or "").strip() or None
                raw_issues = q.get("issues")
                issues = (
                    [str(i).strip() for i in raw_issues if str(i).strip()]
                    if isinstance(raw_issues, list)
                    else None
                )

                tq = TranslationQuality(
                    status="ok" if translation else "failed",
                    score=score,
                    summary=summary,
                    issues=issues,
                    model=self.deployment_name,
                )

                return {
                    "text": translation,
                    "register": register,
                    "localization_notes": localization_notes,
                    "translation_quality": {
                        "status": tq.status,
                        "score": tq.score,
                        "summary": tq.summary,
                        "issues": tq.issues,
                        "model": tq.model,
                    },
                }
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    sleep_s = _openai_retry_sleep_seconds(attempt)
                    logger.warning(
                        "Azure OpenAI rate-limited; sleeping %.2fs",
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                break

        raise RuntimeError(
            "Azure OpenAI translate_with_context_and_quality failed: "
            f"{last_err}"
        )


class GroqTranslator(AbstractTranslator):
    """Free translation using Groq's LLM API.
    
    Groq offers a free tier with fast inference.
    Requires GROQ_API_KEY environment variable.
    
    Sign up at: https://console.groq.com
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str = None,
    ):
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError(
                "groq package not installed. "
                "Install with: pip install groq"
            )

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Sign up for free at https://console.groq.com"
            )

        self.model = model
        # Disable Groq SDK internal retries if supported.
        # We implement retries ourselves so *every* attempt passes through our
        # limiter.
        try:
            self.client = Groq(api_key=self.api_key, max_retries=0)
        except TypeError:
            self.client = Groq(api_key=self.api_key)

        # Groq free tier can rate-limit for many segment calls.
        # These knobs keep the pipeline robust without requiring callers to
        # manage retries.
        self._min_delay_seconds = float(
            os.environ.get("GROQ_MIN_DELAY_SECONDS", "0.3")
        )

        # Hard RPM cap (requests/minute). This is the most common reason for
        # 429s when translating many short segments.
        # Set GROQ_RPM to your account's limit (example: 30). Set to 0 to
        # disable.
        self._rpm = int(os.environ.get("GROQ_RPM", "30"))
        self._rpm_window_seconds = float(
            os.environ.get("GROQ_RPM_WINDOW_SECONDS", "60")
        )

        # A small safety buffer helps with server-side bucket boundaries,
        # network jitter, and clock differences.
        self._rpm_safety_seconds = float(
            os.environ.get("GROQ_RPM_SAFETY_SECONDS", "0.5")
        )
        self._effective_rpm = int(
            os.environ.get("GROQ_EFFECTIVE_RPM", "0")
        )

        self._max_retries = int(os.environ.get("GROQ_MAX_RETRIES", "8"))
        self._backoff_initial_seconds = float(
            os.environ.get("GROQ_BACKOFF_INITIAL_SECONDS", "2.5")
        )
        self._backoff_max_seconds = float(
            os.environ.get("GROQ_BACKOFF_MAX_SECONDS", "30.0")
        )

        _GROQ_LIMITER.configure(
            rpm=self._rpm,
            window_seconds=self._rpm_window_seconds,
            min_delay_seconds=self._min_delay_seconds,
            safety_seconds=self._rpm_safety_seconds,
            effective_rpm=self._effective_rpm,
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        if "ratelimit" in name or "rate limit" in msg:
            return True
        if "429" in msg or "too many requests" in msg:
            return True
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        return False

    def _call_with_retries(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            # Proactively respect RPM + minimum spacing between calls.
            _GROQ_LIMITER.acquire()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                if (
                    (not self._is_rate_limit_error(e))
                    or attempt >= self._max_retries
                ):
                    logger.error("Groq translation failed: %s", str(e))
                    raise

                # Exponential backoff with small jitter.
                delay = min(
                    self._backoff_max_seconds,
                    self._backoff_initial_seconds * (2 ** attempt),
                )
                delay += random.uniform(0.0, min(1.0, 0.25 * delay))
                logger.warning(
                    "Groq rate limited (429). Retrying in %.2fs (%d/%d)",
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)

        # Should never reach here, but keep type-checkers happy.
        raise last_exc if last_exc else RuntimeError("Groq translation failed")

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt = (
            f"Translate from {source_language} to {target_language}. "
            f"Preserve meaning, punctuation, and named entities. "
            f"Output only the translated text:\n\n{text}"
        )

        logger.info("Sending translation to Groq model %s", self.model)
        return self._call_with_retries(prompt)

    def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
        register_hint: Optional[str] = None,
    ) -> str:
        register = infer_register_from_asr(text)
        if register_hint in {"colloquial", "formal", "neutral"}:
            register = register_hint  # type: ignore[assignment]

        prompt = build_context_aware_translation_prompt(
            source_language=source_language,
            target_language=target_language,
            current_text=text,
            previous_segments=list(previous_segments or []),
            next_segments=list(next_segments or []),
            register=register,
        )

        logger.info(
            "Sending context-aware translation to Groq model %s",
            self.model,
        )
        return self._call_with_retries(prompt)


class OllamaTranslator(AbstractTranslator):
    """Local translation using Ollama (completely free, runs offline).
    
    Requires Ollama to be installed and running locally.
    Download from: https://ollama.ai
    
    Example setup:
    1. Install Ollama
    2. Run: ollama pull mistral
    3. Run: ollama serve
    4. Set OLLAMA_BASE_URL if not using default (http://localhost:11434)
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = None,
    ):
        try:
            from ollama import Client
        except ImportError:
            raise RuntimeError(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )

        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.model = model
        self.client = Client(host=self.base_url)

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt = (
            f"Translate from {source_language} to {target_language}. "
            f"Preserve meaning, punctuation, and named entities. "
            f"Output only the translated text:\n\n{text}"
        )

        logger.info("Sending translation to Ollama model %s", self.model)
        try:
            resp = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            out = resp.get("response", "").strip()
            return out
        except Exception as e:
            logger.error("Ollama translation failed: %s", str(e))
            raise

    def translate_with_context(
        self,
        text: str,
        source_language: str,
        target_language: str,
        previous_segments: Optional[Sequence[str]] = None,
        next_segments: Optional[Sequence[str]] = None,
        register_hint: Optional[str] = None,
    ) -> str:
        register = infer_register_from_asr(text)
        if register_hint in {"colloquial", "formal", "neutral"}:
            register = register_hint  # type: ignore[assignment]

        prompt = build_context_aware_translation_prompt(
            source_language=source_language,
            target_language=target_language,
            current_text=text,
            previous_segments=list(previous_segments or []),
            next_segments=list(next_segments or []),
            register=register,
        )

        logger.info(
            "Sending context-aware translation to Ollama model %s",
            self.model,
        )
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
        )
        return resp.get("response", "").strip()
