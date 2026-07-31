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
