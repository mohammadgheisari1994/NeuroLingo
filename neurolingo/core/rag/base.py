"""Abstract interfaces for the RAG subsystem."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Converts raw text into a fixed-size float32 vector."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimensionality."""
        ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float32 array of length self.dim."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Stack individual embeddings into a (N, dim) matrix."""
        return np.stack([self.embed(t) for t in texts]).astype(np.float32)


class VectorStore(ABC):
    """Persistent store for embedding vectors and their associated documents."""

    @abstractmethod
    def add(
        self,
        vector: np.ndarray,
        text: str,
        metadata: dict | None = None,
    ) -> str:
        """Index one document; return its assigned ID."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Return the top_k most similar documents as dicts:
            {"id": str, "text": str, "similarity": float, "metadata": dict}
        Documents below min_similarity are excluded.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of indexed documents."""
        ...

    def save(self) -> None:
        """Persist the index to disk. No-op for in-memory stores."""

    def load(self) -> None:
        """Load the index from disk. No-op for in-memory stores."""
