"""
NumpyVectorStore — a cross-platform, dependency-free vector index.

Uses pure numpy cosine similarity. No FAISS, no C++ builds, no PyTorch.
Vectors are persisted as a .npy binary (float32 matrix) plus a sidecar
.json for document text and metadata — human-readable and easily inspected.

Performance: adequate for educational use (< 10k documents). For 1000 docs
with 4096-dim vectors the full similarity search takes ~1 ms on a modern CPU.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from logger_config import get_logger
from neurolingo.core.rag.base import VectorStore

_log = get_logger(__name__)


@dataclass
class _Document:
    id: str
    text: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)


class NumpyVectorStore(VectorStore):
    """
    In-memory cosine-similarity vector store with optional disk persistence.

    Args:
        persist_path: If given, vectors are saved to <path>.npy and
                      document metadata to <path>.json after every add().
                      Pass None for a purely in-memory, test-friendly store.
    """

    def __init__(self, persist_path: Path | str | None = None) -> None:
        self._docs: list[_Document] = []
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        if self._persist_path:
            self.load()
        _log.debug(
            "NumpyVectorStore initialised | docs=%d | persist=%s",
            len(self._docs), self._persist_path,
        )

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def vector_dim(self) -> int | None:
        """Dimensionality of the stored vectors, or None if the store is
        empty (there's nothing to infer a dimension from yet)."""
        return int(self._docs[0].vector.shape[0]) if self._docs else None

    def clear(self) -> None:
        """
        Remove every indexed document and delete any persisted files.

        Used when switching embedding providers to a different vector
        dimension — old vectors of dimension A can't be compared against a
        new query of dimension B, so the store must be wiped and re-seeded
        rather than silently producing a shape-mismatch error later.
        """
        self._docs = []
        if self._persist_path:
            self._persist_path.with_suffix(".npy").unlink(missing_ok=True)
            self._persist_path.with_suffix(".json").unlink(missing_ok=True)
        _log.info("VectorStore cleared | path=%s", self._persist_path)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(
        self,
        vector: np.ndarray,
        text: str,
        metadata: dict | None = None,
    ) -> str:
        doc_id = str(uuid.uuid4())
        self._docs.append(_Document(
            id=doc_id,
            text=text,
            vector=vector.astype(np.float32),
            metadata=metadata or {},
        ))
        if self._persist_path:
            self.save()
        _log.debug("Document indexed | id=%s | total=%d", doc_id, len(self._docs))
        return doc_id

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Return top_k documents ranked by cosine similarity.

        Cosine similarity = (A · B) / (‖A‖ · ‖B‖)
        Computed in one vectorised pass: normalise both sides, then dot-product.
        """
        if not self._docs:
            return []

        # Build (N, dim) matrix from stored vectors
        matrix = np.stack([d.vector for d in self._docs])  # float32, (N, dim)

        # Normalise query
        q_norm = np.linalg.norm(query_vector)
        q = query_vector.astype(np.float32) / (q_norm + 1e-9)

        # Normalise rows of the matrix
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # (N, 1)
        normed = matrix / (row_norms + 1e-9)

        # Cosine similarities: (N,)
        similarities = normed @ q

        # Select top_k (unsorted first for speed, then sort the subset)
        k = min(top_k, len(self._docs))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < min_similarity:
                continue
            doc = self._docs[int(idx)]
            results.append({
                "id": doc.id,
                "text": doc.text,
                "similarity": sim,
                "metadata": doc.metadata,
            })

        if results:
            _log.debug(
                "RAG retrieved %d contexts with top similarity %.4f",
                len(results), results[0]["similarity"],
            )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._persist_path is None or not self._docs:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Vectors → binary float32
        vecs = np.stack([d.vector for d in self._docs]).astype(np.float32)
        np.save(self._persist_path.with_suffix(".npy"), vecs)
        # Text + metadata → JSON sidecar (human-readable)
        meta = {
            "ids": [d.id for d in self._docs],
            "texts": [d.text for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }
        self._persist_path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log.debug("VectorStore saved | docs=%d | path=%s", len(self._docs), self._persist_path)

    def load(self) -> None:
        npy = self._persist_path.with_suffix(".npy")
        jsn = self._persist_path.with_suffix(".json")
        if not npy.exists() or not jsn.exists():
            return
        vecs = np.load(npy).astype(np.float32)
        meta = json.loads(jsn.read_text(encoding="utf-8"))
        self._docs = [
            _Document(
                id=meta["ids"][i],
                text=meta["texts"][i],
                vector=vecs[i],
                metadata=meta["metadatas"][i],
            )
            for i in range(len(meta["ids"]))
        ]
        _log.info("VectorStore loaded | docs=%d | path=%s", len(self._docs), self._persist_path)
