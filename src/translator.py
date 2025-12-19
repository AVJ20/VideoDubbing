import os
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Literal

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AzureOpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    AzureOpenAI = None


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
                "For free alternatives, see GroqTranslator or OllamaTranslator."
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
            "Sending context-aware translation request to OpenAI model %s",
            self.model,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()


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

        self.api_version = api_version
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
            "Sending context-aware translation to Azure OpenAI deployment %s",
            self.deployment_name,
        )
        resp = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()


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
        self.client = Groq(api_key=self.api_key)

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
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            out = resp.choices[0].message.content.strip()
            return out
        except Exception as e:
            logger.error("Groq translation failed: %s", str(e))
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
            "Sending context-aware translation to Groq model %s",
            self.model,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()


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
