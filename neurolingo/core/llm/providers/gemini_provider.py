"""Google Gemini cloud LLM provider."""
from __future__ import annotations

import asyncio

from logger_config import get_logger
from neurolingo.core.llm.base import (
    AuthenticationError,
    LLMConfig,
    LLMProvider,
    ProviderError,
    ProviderUnavailableError,
)

_log = get_logger(__name__)

try:
    import google.generativeai as _genai

    _AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore[assignment]
    _AVAILABLE = False


class GeminiProvider(LLMProvider):

    def __init__(self, config: LLMConfig) -> None:
        self._key = config.gemini_api_key
        self._model_name = config.gemini_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._model = None

    @property
    def name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        return _AVAILABLE and bool(self._key)

    def _get_model(self):
        if self._model is None:
            _genai.configure(api_key=self._key)
            self._model = _genai.GenerativeModel(
                self._model_name,
                generation_config=_genai.GenerationConfig(
                    max_output_tokens=self._max_tokens,
                    temperature=self._temperature,
                ),
            )
        return self._model

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        full = f"{system}\n\n{prompt}".strip() if system else prompt
        try:
            model = self._get_model()
            response = await asyncio.to_thread(model.generate_content, full)
            return response.text
        except Exception as exc:
            msg = str(exc).lower()
            if "api key" in msg or "authentication" in msg or "permission" in msg:
                raise AuthenticationError(f"Gemini auth failed: {exc}") from exc
            if "timeout" in msg or "connect" in msg or "unavailable" in msg:
                raise ProviderUnavailableError(f"Gemini unavailable: {exc}") from exc
            raise ProviderError(f"Gemini unexpected error: {exc}") from exc

    async def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        # Gemini's GenerativeModel supports multi-turn via history; flatten to text
        # for now and reconstruct as a single prompt.
        parts = []
        if system:
            parts.append(f"[System]: {system}")
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"[{role}]: {m['content']}")
        return await self.generate_explanation("\n".join(parts))
