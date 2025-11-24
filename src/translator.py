import os
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


class AbstractTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        pass


class OpenAITranslator(AbstractTranslator):
    """Translate text using OpenAI chat completion.

    Requires environment variable OPENAI_API_KEY set.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        if openai is None:
            raise RuntimeError("openai package not installed; install requirements to use OpenAITranslator")
        self.model = model

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        prompt = (
            f"You are a helpful translator. Translate the following from {source_language} to "
            f"{target_language} while preserving meaning, punctuation and named entities.\n\nText:\n{text}"
        )

        logger.info("Sending translation request to OpenAI model %s", self.model)
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        out = resp["choices"][0]["message"]["content"].strip()
        return out


class IdentityTranslator(AbstractTranslator):
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        logger.warning("Using IdentityTranslator: returning original text")
        return text
