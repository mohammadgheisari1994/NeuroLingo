"""Google Gemini cloud LLM provider, built on the google-genai SDK.

google-generativeai (the old SDK) reached end-of-life and no longer receives
updates; google-genai is its supported successor and also natively supports
system instructions instead of the old hand-rolled text concatenation.
"""
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
    from google import genai as _genai
    from google.genai import errors as _genai_errors
    from google.genai import types as _genai_types

    _AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore[assignment]
    _genai_errors = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]
    _AVAILABLE = False

_AUTH_STATUS_CODES = {401, 403}
_AUTH_ERROR_REASONS = {"API_KEY_INVALID", "PERMISSION_DENIED"}
_RATE_LIMIT_STATUS_CODE = 429


def _is_auth_error(exc: "_genai_errors.ClientError") -> bool:
    """Gemini reports a bad API key as HTTP 400 INVALID_ARGUMENT, not 401/403 —
    the real signal is a nested `reason` in the error body, so status codes
    alone would misclassify it as a generic ProviderError."""
    if exc.code in _AUTH_STATUS_CODES:
        return True
    try:
        for detail in exc.details.get("error", {}).get("details", []):
            if detail.get("reason") in _AUTH_ERROR_REASONS:
                return True
    except AttributeError:
        pass
    return False


class GeminiProvider(LLMProvider):

    def __init__(self, config: LLMConfig) -> None:
        self._key = config.gemini_api_key
        self._model_name = config.gemini_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._client = None  # lazy-initialised

    @property
    def name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        return _AVAILABLE and bool(self._key)

    def _get_client(self):
        if self._client is None:
            self._client = _genai.Client(api_key=self._key)
        return self._client

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        return await self._generate(prompt, system)

    async def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        # Flatten multi-turn history to a single prompt for now — the router
        # treats every provider as single-shot today, and system instructions
        # are passed natively via config rather than concatenated as text.
        parts = [
            f"[{'User' if m['role'] == 'user' else 'Assistant'}]: {m['content']}"
            for m in messages
        ]
        return await self._generate("\n".join(parts), system)

    async def _generate(self, prompt: str, system: str) -> str:
        try:
            client = self._get_client()
            config = _genai_types.GenerateContentConfig(
                max_output_tokens=self._max_tokens,
                temperature=self._temperature,
                system_instruction=system or None,
            )
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except _genai_errors.ClientError as exc:
            if _is_auth_error(exc):
                raise AuthenticationError(f"Gemini auth failed: {exc}") from exc
            if exc.code == _RATE_LIMIT_STATUS_CODE:
                raise ProviderUnavailableError(f"Gemini rate-limited: {exc}") from exc
            raise ProviderError(f"Gemini client error: {exc}") from exc
        except _genai_errors.ServerError as exc:
            raise ProviderUnavailableError(f"Gemini server error: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Gemini unexpected error: {exc}") from exc
