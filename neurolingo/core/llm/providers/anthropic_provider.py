"""Anthropic (Claude) cloud LLM provider."""
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
    import anthropic as _anthropic

    _AVAILABLE = True
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _AVAILABLE = False


class AnthropicProvider(LLMProvider):

    def __init__(self, config: LLMConfig) -> None:
        self._key = config.anthropic_api_key
        self._model = config.anthropic_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._client = None

    @property
    def name(self) -> str:
        return "Anthropic"

    def is_available(self) -> bool:
        return _AVAILABLE and bool(self._key)

    def _get_client(self):
        if self._client is None:
            self._client = _anthropic.Anthropic(api_key=self._key)
        return self._client

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        return await self.chat([{"role": "user", "content": prompt}], system=system)

    async def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        try:
            client = self._get_client()
            kwargs = dict(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
            )
            if system:
                kwargs["system"] = system
            response = await asyncio.to_thread(client.messages.create, **kwargs)
            return response.content[0].text
        except _anthropic.AuthenticationError as exc:
            raise AuthenticationError(f"Anthropic auth failed: {exc}") from exc
        except (_anthropic.APIConnectionError, _anthropic.APITimeoutError) as exc:
            raise ProviderUnavailableError(f"Anthropic connection error: {exc}") from exc
        except _anthropic.RateLimitError as exc:
            raise ProviderUnavailableError(f"Anthropic rate-limited: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic unexpected error: {exc}") from exc
