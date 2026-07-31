"""
Local llama-cpp-python provider — the offline fallback.

Uses tiny GGUF models (e.g. Phi-3-mini-4k-instruct.Q4_K_M.gguf, ~2 GB)
that run on CPU on all platforms including mobile (via llama.cpp).
The import is fully guarded so the class can be instantiated even when
llama-cpp-python is not installed; is_available() will return False.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from logger_config import get_logger
from neurolingo.core.llm.base import LLMConfig, LLMProvider, ProviderError, ProviderUnavailableError

_log = get_logger(__name__)

try:
    from llama_cpp import Llama as _Llama

    _AVAILABLE = True
except ImportError:
    _Llama = None  # type: ignore[assignment,misc]
    _AVAILABLE = False


class LocalLlamaProvider(LLMProvider):
    """
    Offline LLM provider backed by a local GGUF model file.

    The model is loaded lazily on first call to avoid startup latency.
    Use n_ctx=2048 as the default context window — small enough for
    low-RAM mobile devices.
    """

    def __init__(self, config: LLMConfig, n_ctx: int = 2048, n_threads: int = 4) -> None:
        self._model_path = config.local_model_path
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._llm = None   # lazy-loaded

    @property
    def name(self) -> str:
        return "LocalLlama"

    def is_available(self) -> bool:
        return _AVAILABLE and bool(self._model_path) and Path(self._model_path).is_file()

    def _load(self) -> None:
        if self._llm is not None:
            return
        _log.info("Loading local Llama model | path=%s", self._model_path)
        try:
            self._llm = _Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
            _log.info("Local Llama model loaded successfully")
        except Exception as exc:
            raise ProviderUnavailableError(f"Failed to load local model: {exc}") from exc

    def _format_prompt(self, prompt: str, system: str) -> str:
        """Llama-3 / Mistral instruct format."""
        if system:
            return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
        return f"<s>[INST] {prompt} [/INST]"

    async def generate_explanation(self, prompt: str, system: str = "") -> str:
        try:
            await asyncio.to_thread(self._load)
            full_prompt = self._format_prompt(prompt, system)
            result = await asyncio.to_thread(
                self._llm,
                full_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                echo=False,
            )
            return result["choices"][0]["text"].strip()
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderError(f"LocalLlama inference error: {exc}") from exc

    async def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        # Flatten multi-turn to a single instruct prompt
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        return await self.generate_explanation(history, system=system)
