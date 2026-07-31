"""
Embedding providers for the RAG system.

Mobile / offline constraint: NO PyTorch, NO sentence-transformers, NO FAISS.

Two implementations:
  HashingEmbeddingProvider — 100% offline, pure numpy, deterministic.
      Uses the hashing trick (MD5 for cross-process stability) to map
      text tokens → fixed-size float32 vectors without a vocabulary.

  APIEmbeddingProvider — cloud-backed (OpenAI text-embedding-3-small).
      Falls back gracefully if the key is absent.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from logger_config import get_logger
from neurolingo.core.rag.base import EmbeddingProvider

_log = get_logger(__name__)

try:
    import openai as _openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _openai = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False


# ── Offline fallback ──────────────────────────────────────────────────────────

class HashingEmbeddingProvider(EmbeddingProvider):
    """
    Offline embedding via the feature-hashing trick (aka hashing vectoriser).

    Algorithm:
      1. Tokenise text into unigrams + bigrams (lower-cased).
      2. Map each token to a bucket in [0, n_features) via MD5 (deterministic
         across processes unlike Python's built-in hash()).
      3. Accumulate term frequencies.
      4. L2-normalise the result to unit length.

    Properties:
      - Fixed output dimension regardless of vocabulary.
      - Deterministic across processes and platforms.
      - Zero external dependencies beyond numpy.
      - Suitable for cosine-similarity search.

    Trade-off: hash collisions degrade recall slightly for very large n_features.
    Default of 4096 gives good recall for short educational texts.
    """

    def __init__(self, n_features: int = 4096) -> None:
        self._n = n_features
        _log.debug("HashingEmbeddingProvider initialised | n_features=%d", n_features)

    @property
    def dim(self) -> int:
        return self._n

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._n, dtype=np.float32)
        for token in self._tokenize(text):
            idx = self._hash(token)
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    # ── Internals ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        unigrams = words
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return unigrams + bigrams

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self._n


# ── Cloud-backed provider ─────────────────────────────────────────────────────

class APIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI text-embedding-3-small (1536-dim vectors).

    Falls back gracefully: is_available() returns False if the openai
    package is not installed or no API key is set, allowing the caller
    to switch to HashingEmbeddingProvider.
    """

    _MODEL = "text-embedding-3-small"
    _DIM = 1536

    def __init__(self, api_key: str = "") -> None:
        self._key = api_key
        self._client = None
        _log.debug("APIEmbeddingProvider initialised | available=%s", self.is_available())

    @property
    def dim(self) -> int:
        return self._DIM

    def is_available(self) -> bool:
        return _OPENAI_AVAILABLE and bool(self._key)

    def _get_client(self):
        if self._client is None:
            self._client = _openai.OpenAI(api_key=self._key)
        return self._client

    def embed(self, text: str) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=text, model=self._MODEL)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        # API returns unit vectors, but normalise defensively
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec


# ── Factory ───────────────────────────────────────────────────────────────────

def build_embedding_provider(openai_api_key: str = "") -> EmbeddingProvider:
    """
    Prefer real embeddings (meaningfully better retrieval quality) when a
    usable OpenAI key is configured; fall back to the offline hashing trick
    otherwise — the same cloud-then-offline pattern LLMRouter already uses
    for text generation.

    Note: is_available() only checks that the key is present and the openai
    package importable, not that the key is actually valid/reachable — same
    as every LLMProvider. Callers that use the returned provider to seed a
    knowledge base at startup should handle embedding failures gracefully
    rather than let a bad/unreachable key crash the whole app.
    """
    api_provider = APIEmbeddingProvider(api_key=openai_api_key)
    if api_provider.is_available():
        _log.info("Using APIEmbeddingProvider (OpenAI) for RAG embeddings")
        return api_provider
    _log.info("Using HashingEmbeddingProvider (offline) for RAG embeddings")
    return HashingEmbeddingProvider()
