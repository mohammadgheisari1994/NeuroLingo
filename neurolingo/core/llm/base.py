"""
Abstract interfaces and shared types for the LLM subsystem.

Keeping exceptions, config, and the ABC in one file means every provider
imports exactly one thing and stays fully decoupled from every other provider.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ── Exceptions ────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Base class for all LLM provider failures."""


class AuthenticationError(ProviderError):
    """API key is absent, invalid, or expired."""


class ProviderUnavailableError(ProviderError):
    """Provider is unreachable: network error, timeout, or rate-limit."""


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    User-supplied router configuration.

    Populate via LLMConfig.from_env() at startup, or build manually from
    the Settings UI once that module is implemented.
    """

    preferred_provider: str = "anthropic"   # "openai" | "anthropic" | "gemini" | "local"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    local_model_path: str = ""             # absolute path to a GGUF model file
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-1.5-flash"
    max_tokens: int = 512
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Read keys and preferences from environment / .env file."""
        return cls(
            preferred_provider=os.getenv("LLM_PREFERRED", "anthropic"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            local_model_path=os.getenv("LOCAL_MODEL_PATH", ""),
        )


# ── Abstract provider ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Common async interface for all LLM backends.

    Implementations must handle their own ImportError for optional deps so
    that the provider object can always be constructed — is_available() will
    return False when the underlying library is missing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in log messages."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if the provider can accept requests right now
        (API key set, model file present, library installed, etc.).
        """
        ...

    @abstractmethod
    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        """
        Single-turn text generation.

        Raises:
            AuthenticationError: bad or missing API key.
            ProviderUnavailableError: network / timeout / rate-limit.
            ProviderError: any other provider-specific failure.
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        """
        Multi-turn conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            system: Optional system instruction prepended to the conversation.

        Raises:
            AuthenticationError / ProviderUnavailableError / ProviderError (same as above).
        """
        ...
