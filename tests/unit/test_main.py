"""
Unit tests for the pure-logic helpers in main.py.

main.py's UI class (NeuroLingoApp) needs a live Flet Page and isn't exercised
here; these tests cover the framework-independent wiring — specifically that
_seed_knowledge() actually grounds the AI tutor's RAG store in something
retrievable, which is the whole point of wiring RAGManager into the app.
"""
from __future__ import annotations

import main
from neurolingo.core.llm.router import LLMRouter
from neurolingo.core.rag.embeddings import HashingEmbeddingProvider
from neurolingo.core.rag.rag_manager import RAGManager
from neurolingo.core.rag.vectorstore import NumpyVectorStore


def _build_rag(min_similarity: float = 0.5) -> tuple[RAGManager, NumpyVectorStore]:
    store = NumpyVectorStore(persist_path=None)
    router = LLMRouter(providers=[], local_provider=None)
    rag = RAGManager(
        store, HashingEmbeddingProvider(), router, min_similarity=min_similarity
    )
    return rag, store


def test_seed_knowledge_indexes_one_document_per_sample():
    rag, store = _build_rag()

    main._seed_knowledge(rag)

    assert len(store) == len(main._SAMPLES)


def test_seed_knowledge_content_is_retrievable():
    rag, _store = _build_rag(min_similarity=0.0)

    main._seed_knowledge(rag)

    en, _fa, notes = main._SAMPLES[0]
    query = f"{en} — {notes}"
    results = rag.retrieve(query, top_k=1)

    assert results
    assert results[0]["text"] == query
    assert results[0]["similarity"] > 0.99


# ── _prepare_vector_store (embedding-provider dimension safety) ──────────────

def test_prepare_vector_store_keeps_matching_dimension_data(tmp_path):
    path = tmp_path / "knowledge"
    seed_store = NumpyVectorStore(persist_path=path)
    seed_store.add(HashingEmbeddingProvider().embed("hello"), "hello")

    store = main._prepare_vector_store(path, HashingEmbeddingProvider())

    assert len(store) == 1  # same dimension -> data is kept, not wiped


def test_prepare_vector_store_wipes_mismatched_dimension_data(tmp_path):
    path = tmp_path / "knowledge"
    old_embedder = HashingEmbeddingProvider(n_features=64)
    seed_store = NumpyVectorStore(persist_path=path)
    seed_store.add(old_embedder.embed("hello"), "hello")
    assert seed_store.vector_dim == 64

    new_embedder = HashingEmbeddingProvider(n_features=256)
    store = main._prepare_vector_store(path, new_embedder)

    assert len(store) == 0  # dimension changed -> old data wiped, not crashed
    assert not path.with_suffix(".npy").exists()


def test_prepare_vector_store_empty_path_is_fine(tmp_path):
    store = main._prepare_vector_store(tmp_path / "does-not-exist-yet", HashingEmbeddingProvider())
    assert len(store) == 0
