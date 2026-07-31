"""
Unit tests for LLMRouter.

No real API calls are made. All providers are replaced with AsyncMocks so
tests are instant and fully deterministic.

Coverage targets:
- Routing to the preferred available provider
- Fallback on AuthenticationError
- Fallback on ProviderUnavailableError
- Fallback on generic ProviderError
- Skipping unavailable (is_available=False) providers
- Raising when no provider at all is available
- chat() delegates through the same routing path
- Routing log messages (smoke-check via caplog)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neurolingo.core.llm.base import (
    AuthenticationError,
    LLMConfig,
    LLMProvider,
    ProviderError,
    ProviderUnavailableError,
)
from neurolingo.core.llm.router import LLMRouter

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_provider(
    name: str,
    available: bool = True,
    answer: str = "mocked answer",
    side_effect=None,
) -> AsyncMock:
    """Build a fully-mocked LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = name
    provider.is_available.return_value = available

    if side_effect is not None:
        provider.generate_explanation = AsyncMock(side_effect=side_effect)
        provider.chat = AsyncMock(side_effect=side_effect)
    else:
        provider.generate_explanation = AsyncMock(return_value=answer)
        provider.chat = AsyncMock(return_value=answer)
    return provider


def _router(providers, local=None) -> LLMRouter:
    return LLMRouter(providers=providers, local_provider=local)


@pytest.fixture(autouse=True)
def _no_real_retry_delay(monkeypatch):
    """Retries back off via asyncio.sleep; patch it so every test in this
    file stays instant regardless of how many ProviderUnavailableError
    retries it triggers, instead of only the tests that remember to."""
    monkeypatch.setattr("neurolingo.core.llm.router.asyncio.sleep", AsyncMock())


# ── Routing to preferred provider ─────────────────────────────────────────────

async def test_calls_first_available_provider():
    p1 = _make_provider("P1", answer="from P1")
    router = _router([p1])
    assert await router.generate_explanation("hello") == "from P1"
    p1.generate_explanation.assert_called_once()


async def test_returns_first_provider_response_without_calling_others():
    p1 = _make_provider("P1", answer="first")
    p2 = _make_provider("P2", answer="second")
    router = _router([p1, p2])
    result = await router.generate_explanation("test")
    assert result == "first"
    p2.generate_explanation.assert_not_called()


# ── Fallback on auth error ────────────────────────────────────────────────────

async def test_falls_back_on_auth_error():
    primary = _make_provider("Cloud", side_effect=AuthenticationError("bad key"))
    local = _make_provider("Local", answer="local answer")
    router = _router([primary], local=local)
    result = await router.generate_explanation("prompt")
    assert result == "local answer"
    local.generate_explanation.assert_called_once()


async def test_falls_back_on_unavailable_error():
    primary = _make_provider("Cloud", side_effect=ProviderUnavailableError("timeout"))
    local = _make_provider("Local", answer="offline")
    router = _router([primary], local=local)
    assert await router.generate_explanation("x") == "offline"


async def test_falls_back_on_generic_provider_error():
    primary = _make_provider("Cloud", side_effect=ProviderError("unknown"))
    local = _make_provider("Local", answer="local")
    router = _router([primary], local=local)
    assert await router.generate_explanation("x") == "local"


# ── Skipping unavailable providers ────────────────────────────────────────────

async def test_skips_unavailable_provider_without_calling_it():
    unavailable = _make_provider("Offline", available=False)
    available = _make_provider("Online", answer="online answer")
    router = _router([unavailable, available])
    result = await router.generate_explanation("test")
    assert result == "online answer"
    unavailable.generate_explanation.assert_not_called()


async def test_multiple_cloud_failures_reach_local():
    p1 = _make_provider("P1", side_effect=AuthenticationError("bad"))
    p2 = _make_provider("P2", side_effect=ProviderUnavailableError("timeout"))
    local = _make_provider("Local", answer="local wins")
    router = _router([p1, p2], local=local)
    assert await router.generate_explanation("x") == "local wins"


# ── No provider available ─────────────────────────────────────────────────────

async def test_raises_when_no_provider_available():
    router = _router([], local=None)
    with pytest.raises(ProviderError, match="No LLM provider"):
        await router.generate_explanation("test")


async def test_raises_when_all_fail_and_no_local():
    cloud = _make_provider("Cloud", side_effect=AuthenticationError("x"))
    router = _router([cloud], local=None)
    with pytest.raises(ProviderError):
        await router.generate_explanation("test")


async def test_raises_when_local_also_fails():
    cloud = _make_provider("Cloud", side_effect=AuthenticationError("x"))
    local = _make_provider("Local", side_effect=ProviderError("disk error"))
    router = _router([cloud], local=local)
    with pytest.raises(ProviderError, match="All providers failed"):
        await router.generate_explanation("test")


# ── Local unavailable ─────────────────────────────────────────────────────────

async def test_skips_local_when_not_available():
    cloud = _make_provider("Cloud", side_effect=AuthenticationError("x"))
    local = _make_provider("Local", available=False)
    router = _router([cloud], local=local)
    with pytest.raises(ProviderError):
        await router.generate_explanation("test")


# ── chat() delegates correctly ────────────────────────────────────────────────

async def test_chat_routes_to_available_provider():
    p = _make_provider("P1", answer="chat reply")
    router = _router([p])
    messages = [{"role": "user", "content": "hello"}]
    result = await router.chat(messages, system="you are helpful")
    assert result == "chat reply"
    p.chat.assert_called_once_with(messages, "you are helpful")


async def test_chat_falls_back_on_auth_error():
    cloud = _make_provider("Cloud")
    cloud.chat = AsyncMock(side_effect=AuthenticationError("bad"))
    local = _make_provider("Local")
    local.chat = AsyncMock(return_value="local chat")
    router = _router([cloud], local=local)
    result = await router.chat([{"role": "user", "content": "hi"}])
    assert result == "local chat"


# ── Retry with backoff on transient failures ──────────────────────────────────

async def test_retries_transient_failure_then_succeeds():
    """A provider that fails twice with ProviderUnavailableError then succeeds
    on the third attempt should return that success WITHOUT falling over to
    another provider — this is the whole point of the retry."""
    flaky = _make_provider(
        "Flaky",
        side_effect=[
            ProviderUnavailableError("timeout 1"),
            ProviderUnavailableError("timeout 2"),
            "recovered",
        ],
    )
    other = _make_provider("Other", answer="should not be used")
    router = _router([flaky, other])

    result = await router.generate_explanation("x")

    assert result == "recovered"
    assert flaky.generate_explanation.call_count == 3
    other.generate_explanation.assert_not_called()


async def test_gives_up_after_max_retries_and_falls_over():
    """More transient failures than the retry budget allows should still
    exhaust the retries and then fall through to the next provider, exactly
    like before retries existed — just with extra attempts first."""
    always_flaky = _make_provider("AlwaysFlaky", side_effect=ProviderUnavailableError("down"))
    other = _make_provider("Other", answer="fallback answer")
    router = _router([always_flaky, other])

    result = await router.generate_explanation("x")

    assert result == "fallback answer"
    # 1 initial attempt + _MAX_RETRIES (2) retries = 3 total calls
    assert always_flaky.generate_explanation.call_count == 3


async def test_auth_error_is_never_retried(monkeypatch):
    """AuthenticationError must fail over immediately — retrying a bad key
    wastes time and can trip provider rate limits for no benefit."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("neurolingo.core.llm.router.asyncio.sleep", sleep_mock)
    bad_key = _make_provider("BadKey", side_effect=AuthenticationError("invalid key"))
    local = _make_provider("Local", answer="local")
    router = _router([bad_key], local=local)

    result = await router.generate_explanation("x")

    assert result == "local"
    bad_key.generate_explanation.assert_called_once()
    sleep_mock.assert_not_called()


# ── LLMConfig.from_env() ──────────────────────────────────────────────────────

def test_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PREFERRED", "anthropic")
    config = LLMConfig.from_env()
    assert config.anthropic_api_key == "sk-test"
    assert config.preferred_provider == "anthropic"


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.preferred_provider == "anthropic"
    assert config.max_tokens == 512
    assert config.temperature == 0.7


# ── Builder ───────────────────────────────────────────────────────────────────

def test_router_requires_config_or_providers():
    with pytest.raises(ValueError, match="config or providers"):
        LLMRouter()  # no args


def test_router_build_from_config_puts_preferred_first():
    config = LLMConfig(preferred_provider="openai", openai_api_key="k")
    router = LLMRouter(config=config)
    assert router._providers[0].name == "OpenAI"


def test_router_build_from_config_all_three_providers():
    config = LLMConfig()
    router = LLMRouter(config=config)
    names = [p.name for p in router._providers]
    assert set(names) == {"Anthropic", "OpenAI", "Gemini"}


# ── Logging ───────────────────────────────────────────────────────────────────

async def test_info_log_on_successful_route(caplog):
    import logging
    p = _make_provider("Anthropic", answer="ok")
    router = _router([p])
    with caplog.at_level(logging.INFO, logger="neurolingo"):
        await router.generate_explanation("test")
    assert "Routing request to Anthropic" in caplog.text


async def test_warning_log_on_auth_fallback(caplog):
    import logging
    cloud = _make_provider("Anthropic", side_effect=AuthenticationError("x"))
    local = _make_provider("Local", answer="ok")
    router = _router([cloud], local=local)
    with caplog.at_level(logging.WARNING, logger="neurolingo"):
        await router.generate_explanation("test")
    assert "authentication error" in caplog.text.lower()
