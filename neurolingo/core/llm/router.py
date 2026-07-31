"""
LLMRouter — central routing logic for all AI generation calls.

Priority order:
  1. Preferred cloud provider (if API key present and provider reachable)
  2. Other configured cloud providers (automatic failover)
  3. LocalLlamaProvider (offline fallback)

Every routing decision — including skips and fallbacks — is logged so the
user's Developer Console shows exactly which provider answered each request.
"""
from __future__ import annotations

from logger_config import get_logger
from neurolingo.core.llm.base import (
    AuthenticationError,
    LLMConfig,
    LLMProvider,
    ProviderError,
    ProviderUnavailableError,
)
from neurolingo.core.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    LocalLlamaProvider,
    OpenAIProvider,
)

_log = get_logger(__name__)

_PROVIDER_ORDER = ["anthropic", "openai", "gemini"]


class LLMRouter:
    """
    Routes LLM requests through cloud providers with automatic failover to a
    local offline model.

    Dependency injection (the `providers` / `local_provider` kwargs) is
    intentionally exposed so unit tests can wire in mocks without patching.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        providers: list[LLMProvider] | None = None,
        local_provider: LLMProvider | None = None,
    ) -> None:
        if providers is not None:
            # DI path — used in tests
            self._providers = providers
            self._local = local_provider
        elif config is not None:
            self._providers = self._build_cloud_providers(config)
            self._local = LocalLlamaProvider(config)
        else:
            raise ValueError("Provide either config or providers for dependency injection.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        """Route a single-turn request through the provider chain."""
        return await self._route(
            lambda p: p.generate_explanation(prompt, system),
            context=f"prompt='{prompt[:60]}…'" if len(prompt) > 60 else f"prompt='{prompt}'",
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        """Route a multi-turn chat request through the provider chain."""
        return await self._route(
            lambda p: p.chat(messages, system),
            context=f"turns={len(messages)}",
        )

    # ── Internal routing ──────────────────────────────────────────────────────

    async def _route(self, call, context: str) -> str:
        """
        Try each cloud provider in order; fall back to local on any failure.
        `call` is a coroutine factory: lambda p: p.some_method(...)
        """
        for provider in self._providers:
            if not provider.is_available():
                _log.debug("Skipping %s — not available (%s)", provider.name, context)
                continue

            try:
                _log.info("Routing request to %s | %s", provider.name, context)
                result = await call(provider)
                _log.debug("Response received from %s", provider.name)
                return result

            except AuthenticationError as exc:
                _log.warning(
                    "%s authentication error (%s). Trying next provider.", provider.name, exc
                )
            except ProviderUnavailableError as exc:
                _log.warning(
                    "%s API failed (%s). Falling back to next provider.", provider.name, exc
                )
            except ProviderError as exc:
                _log.warning(
                    "%s unexpected error (%s). Trying next provider.", provider.name, exc
                )

        # All cloud providers failed — try local
        if self._local is not None and self._local.is_available():
            _log.warning(
                "All cloud providers failed or unavailable. Falling back to %s.",
                self._local.name,
            )
            try:
                return await call(self._local)
            except Exception as exc:
                _log.exception("Local fallback also failed: %s", exc)
                raise ProviderError(f"All providers failed. Last error: {exc}") from exc

        raise ProviderError(
            "No LLM provider is available. "
            "Configure an API key in Settings or install llama-cpp-python with a GGUF model."
        )

    # ── Builder ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_cloud_providers(config: LLMConfig) -> list[LLMProvider]:
        """
        Instantiate all three cloud providers, preferred provider first.
        Providers with empty API keys will report is_available()=False and
        be skipped at call time — no eager validation needed.
        """
        all_providers: dict[str, LLMProvider] = {
            "anthropic": AnthropicProvider(config),
            "openai": OpenAIProvider(config),
            "gemini": GeminiProvider(config),
        }
        ordered: list[LLMProvider] = []
        preferred = config.preferred_provider
        if preferred in all_providers:
            ordered.append(all_providers.pop(preferred))
        # Remaining providers in deterministic order
        for name in _PROVIDER_ORDER:
            if name in all_providers:
                ordered.append(all_providers[name])
        return ordered
