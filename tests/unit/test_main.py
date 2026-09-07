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


# ── _shuffled_scramble_pool (#52 word-scramble review puzzle) ────────────────

def test_shuffled_scramble_pool_is_a_permutation_of_the_input():
    words = ["She", "has", "been", "waiting", "for", "the", "bus."]
    pool = main._shuffled_scramble_pool(words)
    assert sorted(pool) == sorted(words)
    assert pool is not words  # never mutates the caller's list


def test_shuffled_scramble_pool_never_matches_original_order():
    # Run many times since random.shuffle() could otherwise coincidentally
    # reproduce the original order — the anti-identity guard must catch it
    # every time, not just usually.
    words = ["One", "two"]
    for _ in range(200):
        assert main._shuffled_scramble_pool(words) != words


def test_shuffled_scramble_pool_single_word_is_unchanged():
    assert main._shuffled_scramble_pool(["Hello."]) == ["Hello."]


def test_shuffled_scramble_pool_empty_list_is_fine():
    assert main._shuffled_scramble_pool([]) == []


# ── _build_session_summary (#54 end-of-session recap) ────────────────────────
#
# Reads only plain int/bool instance attributes — no Flet Page needed — so a
# bare object with just those attributes set is enough to call the real
# bound method directly.

def _fake_session(count, first_try_correct, first_try_total, best_streak):
    app = object.__new__(main.NeuroLingoApp)
    app._session_count = count
    app._session_first_try_correct = first_try_correct
    app._session_first_try_total = first_try_total
    app._session_best_streak = best_streak
    return app


def test_session_summary_is_empty_when_no_cards_were_reviewed():
    # The queue was already empty on arrival — no session happened, so no
    # summary should be shown (distinct from "reviewed 0 cards").
    app = _fake_session(count=0, first_try_correct=0, first_try_total=0, best_streak=0)
    assert main.NeuroLingoApp._build_session_summary(app) == ""


def test_session_summary_reports_counts_and_accuracy():
    app = _fake_session(count=5, first_try_correct=4, first_try_total=5, best_streak=3)
    summary = main.NeuroLingoApp._build_session_summary(app)
    assert "5 reviewed" in summary
    assert "80% first-try" in summary
    assert "best streak 3" in summary


def test_session_summary_handles_perfect_and_zero_accuracy():
    perfect = _fake_session(count=2, first_try_correct=2, first_try_total=2, best_streak=2)
    assert "100% first-try" in main.NeuroLingoApp._build_session_summary(perfect)

    zero = _fake_session(count=2, first_try_correct=0, first_try_total=2, best_streak=0)
    assert "0% first-try" in main.NeuroLingoApp._build_session_summary(zero)
