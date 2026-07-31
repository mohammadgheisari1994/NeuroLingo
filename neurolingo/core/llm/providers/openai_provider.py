"""OpenAI cloud LLM provider."""
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
    import openai as _openai

    _AVAILABLE = True
except ImportError:
    _openai = None  # type: ignore[assignment]
    _AVAILABLE = False


class OpenAIProvider(LLMProvider):

    def __init__(self, config: LLMConfig) -> None:
        self._key = config.openai_api_key
        self._model = config.openai_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._client = None  # lazy-initialised

    @property
    def name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        return _AVAILABLE and bool(self._key)

    def _get_client(self):
        if self._client is None:
            self._client = _openai.OpenAI(api_key=self._key)
        return self._client

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, system="")  # system already embedded

    async def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self._model,
                messages=full_messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except _openai.AuthenticationError as exc:
            raise AuthenticationError(f"OpenAI auth failed: {exc}") from exc
        except (_openai.APIConnectionError, _openai.APITimeoutError) as exc:
            raise ProviderUnavailableError(f"OpenAI connection error: {exc}") from exc
        except _openai.RateLimitError as exc:
            raise ProviderUnavailableError(f"OpenAI rate-limited: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"OpenAI unexpected error: {exc}") from exc
