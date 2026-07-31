"""
Unit tests for the RAG subsystem.

Coverage targets:
  NumpyVectorStore  — cosine similarity correctness, top-k, min_similarity,
                      persistence (save/load), edge cases (empty store, len).
  HashingEmbeddingProvider — unit vectors, determinism, bigrams, batch.
  RAGManager        — add_knowledge → retrieve pipeline; tutor_analyze_mistake
                      prompt construction (LLM mocked).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from neurolingo.core.rag.embeddings import (
    APIEmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider,
)
from neurolingo.core.rag.rag_manager import RAGManager, TutorConversation, _build_tutor_prompt
from neurolingo.core.rag.vectorstore import NumpyVectorStore

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store() -> NumpyVectorStore:
    return NumpyVectorStore()  # in-memory, no persist_path


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(n_features=256)


# ── NumpyVectorStore — basic ──────────────────────────────────────────────────

def test_empty_store_len_is_zero(store):
    assert len(store) == 0


def test_add_increments_len(store):
    store.add(np.array([1.0, 0.0, 0.0], dtype=np.float32), "doc A")
    store.add(np.array([0.0, 1.0, 0.0], dtype=np.float32), "doc B")
    assert len(store) == 2


def test_add_returns_string_id(store):
    doc_id = store.add(np.ones(4, dtype=np.float32), "text")
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_search_empty_store_returns_empty_list(store):
    assert store.search(np.ones(3, dtype=np.float32)) == []


# ── NumpyVectorStore — cosine similarity math ─────────────────────────────────

def test_identical_vector_has_similarity_one(store):
    v = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    store.add(v.copy(), "doc")
    results = store.search(v.copy(), top_k=1)
    assert len(results) == 1
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_have_zero_similarity(store):
    store.add(np.array([1.0, 0.0, 0.0], dtype=np.float32), "x-doc")
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=1)
    assert results[0]["similarity"] == pytest.approx(0.0, abs=1e-5)


def test_returns_most_similar_document_first(store):
    # doc A is along x-axis, doc B is along y-axis
    # query is close to x — doc A should win
    store.add(np.array([1.0, 0.0, 0.0], dtype=np.float32), "doc A")
    store.add(np.array([0.0, 1.0, 0.0], dtype=np.float32), "doc B")
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)
    assert results[0]["text"] == "doc A"
    assert results[1]["text"] == "doc B"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_known_cosine_similarity_value(store):
    # cos([1,1,0], [1,0,0]) = 1/sqrt(2) ≈ 0.7071
    store.add(np.array([1.0, 1.0, 0.0], dtype=np.float32), "diag")
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=1)
    expected = 1.0 / np.sqrt(2)
    assert results[0]["similarity"] == pytest.approx(expected, abs=1e-5)


def test_similarity_is_symmetric(store):
    a = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    b = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    store.add(b.copy(), "b")
    r = store.search(a, top_k=1)
    sim_ab = r[0]["similarity"]

    store2 = NumpyVectorStore()
    store2.add(a.copy(), "a")
    r2 = store2.search(b, top_k=1)
    sim_ba = r2[0]["similarity"]

    assert sim_ab == pytest.approx(sim_ba, abs=1e-5)


# ── NumpyVectorStore — top_k and min_similarity ───────────────────────────────

def test_top_k_is_respected(store):
    for i in range(5):
        v = np.zeros(3, dtype=np.float32)
        v[i % 3] = 1.0
        store.add(v, f"doc {i}")
    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert len(results) == 2


def test_top_k_larger_than_store_returns_all(store):
    store.add(np.array([1.0, 0.0], dtype=np.float32), "only doc")
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=100)
    assert len(results) == 1


def test_min_similarity_filters_low_results(store):
    store.add(np.array([1.0, 0.0, 0.0], dtype=np.float32), "x")
    store.add(np.array([0.0, 1.0, 0.0], dtype=np.float32), "y")
    # query along x — y has similarity ~0, should be filtered
    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=2,
        min_similarity=0.5,
    )
    assert all(r["text"] == "x" for r in results)


def test_result_contains_expected_keys(store):
    store.add(np.ones(4, dtype=np.float32), "text", metadata={"source": "test"})
    result = store.search(np.ones(4, dtype=np.float32), top_k=1)[0]
    assert "id" in result
    assert "text" in result
    assert "similarity" in result
    assert "metadata" in result
    assert result["metadata"]["source"] == "test"


# ── NumpyVectorStore — persistence ────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "vectors"
    store = NumpyVectorStore(persist_path=path)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store.add(v, "persistent doc", metadata={"key": "val"})

    # Load into a fresh store instance
    store2 = NumpyVectorStore(persist_path=path)
    assert len(store2) == 1
    results = store2.search(v, top_k=1)
    assert results[0]["text"] == "persistent doc"
    assert results[0]["metadata"]["key"] == "val"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_save_creates_npy_and_json_files(tmp_path):
    path = tmp_path / "idx"
    store = NumpyVectorStore(persist_path=path)
    store.add(np.ones(4, dtype=np.float32), "hello")
    assert path.with_suffix(".npy").exists()
    assert path.with_suffix(".json").exists()


def test_load_from_nonexistent_path_does_not_raise(tmp_path):
    store = NumpyVectorStore(persist_path=tmp_path / "missing")
    assert len(store) == 0


def test_persisted_json_is_human_readable(tmp_path):
    path = tmp_path / "vs"
    store = NumpyVectorStore(persist_path=path)
    store.add(np.ones(2, dtype=np.float32), "hello world", metadata={"lang": "en"})
    meta = json.loads(path.with_suffix(".json").read_text())
    assert "hello world" in meta["texts"]


# ── NumpyVectorStore — vector_dim / clear (embedding-provider switching) ──────

def test_vector_dim_is_none_when_empty(store):
    assert store.vector_dim is None


def test_vector_dim_reflects_stored_vectors(store):
    store.add(np.ones(7, dtype=np.float32), "doc")
    assert store.vector_dim == 7


def test_clear_empties_an_in_memory_store(store):
    store.add(np.ones(4, dtype=np.float32), "doc")
    store.clear()
    assert len(store) == 0
    assert store.vector_dim is None


def test_clear_deletes_persisted_files(tmp_path):
    path = tmp_path / "idx"
    store = NumpyVectorStore(persist_path=path)
    store.add(np.ones(4, dtype=np.float32), "doc")
    assert path.with_suffix(".npy").exists()

    store.clear()

    assert not path.with_suffix(".npy").exists()
    assert not path.with_suffix(".json").exists()


def test_clear_then_reload_from_disk_stays_empty(tmp_path):
    """A cleared+persisted store must not resurrect old vectors on reload."""
    path = tmp_path / "idx"
    store = NumpyVectorStore(persist_path=path)
    store.add(np.ones(4, dtype=np.float32), "doc")
    store.clear()

    reloaded = NumpyVectorStore(persist_path=path)
    assert len(reloaded) == 0


# ── HashingEmbeddingProvider ──────────────────────────────────────────────────

def test_embed_returns_correct_dimension(embedder):
    vec = embedder.embed("hello world")
    assert vec.shape == (256,)


def test_embed_returns_unit_vector(embedder):
    vec = embedder.embed("the quick brown fox")
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_embed_returns_float32(embedder):
    vec = embedder.embed("test")
    assert vec.dtype == np.float32


def test_embed_is_deterministic(embedder):
    v1 = embedder.embed("deterministic test")
    v2 = embedder.embed("deterministic test")
    np.testing.assert_array_equal(v1, v2)


def test_different_texts_produce_different_vectors(embedder):
    v1 = embedder.embed("the cat sat on the mat")
    v2 = embedder.embed("quantum entanglement phenomena")
    assert not np.allclose(v1, v2)


def test_similar_texts_have_higher_similarity_than_unrelated(embedder):
    v_base = embedder.embed("the cat sat on the mat")
    v_similar = embedder.embed("a cat is sitting on the mat")
    v_unrelated = embedder.embed("quantum physics and dark matter")

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    assert cos(v_base, v_similar) > cos(v_base, v_unrelated)


def test_embed_empty_string_returns_zero_vector(embedder):
    vec = embedder.embed("")
    assert np.all(vec == 0.0)


def test_embed_batch_shape(embedder):
    texts = ["hello", "world", "foo"]
    matrix = embedder.embed_batch(texts)
    assert matrix.shape == (3, 256)


def test_embed_batch_matches_individual_embeds(embedder):
    texts = ["hello world", "good morning"]
    batch = embedder.embed_batch(texts)
    for i, t in enumerate(texts):
        np.testing.assert_array_almost_equal(batch[i], embedder.embed(t))


# ── build_embedding_provider (cloud-then-offline factory) ─────────────────────

def test_build_embedding_provider_uses_hashing_when_no_key():
    provider = build_embedding_provider(openai_api_key="")
    assert isinstance(provider, HashingEmbeddingProvider)


def test_build_embedding_provider_uses_hashing_when_openai_package_missing(monkeypatch):
    import neurolingo.core.rag.embeddings as embeddings_module
    monkeypatch.setattr(embeddings_module, "_OPENAI_AVAILABLE", False)
    provider = build_embedding_provider(openai_api_key="sk-not-actually-checked")
    assert isinstance(provider, HashingEmbeddingProvider)


def test_build_embedding_provider_uses_api_provider_when_key_present(monkeypatch):
    import neurolingo.core.rag.embeddings as embeddings_module
    monkeypatch.setattr(embeddings_module, "_OPENAI_AVAILABLE", True)
    provider = build_embedding_provider(openai_api_key="sk-fake-key-for-test")
    assert isinstance(provider, APIEmbeddingProvider)
    assert provider.dim == 1536


# ── RAGManager ────────────────────────────────────────────────────────────────

@pytest.fixture
def rag_manager(embedder):
    store = NumpyVectorStore()
    mock_router = AsyncMock()
    mock_router.generate_explanation = AsyncMock(return_value="Great explanation!")
    mock_router.chat = AsyncMock(return_value="Great explanation!")
    return RAGManager(
        vector_store=store,
        embedding_provider=embedder,
        llm_router=mock_router,
        min_similarity=0.0,   # always include results in tests
    )


def test_add_knowledge_indexes_document(rag_manager):
    doc_id = rag_manager.add_knowledge("The present perfect tense uses 'have/has'.")
    assert isinstance(doc_id, str)


def test_retrieve_returns_relevant_document(rag_manager):
    rag_manager.add_knowledge("Use 'have been' for present perfect continuous.")
    rag_manager.add_knowledge("Use 'a' before consonant sounds.")
    results = rag_manager.retrieve("present perfect continuous tense")
    assert len(results) > 0
    assert any("present perfect" in r["text"].lower() for r in results)


async def test_tutor_analyze_mistake_calls_llm(rag_manager):
    rag_manager.add_knowledge("Use 'have gone' not 'have went'.")
    result = await rag_manager.tutor_analyze_mistake(
        target_sentence="She has gone to the store.",
        user_input="She have went to the store.",
    )
    assert result == "Great explanation!"
    rag_manager._router.generate_explanation.assert_called_once()


async def test_tutor_analyze_mistake_includes_context_in_prompt(rag_manager):
    """Verify the LLM receives target + user_input in its prompt."""
    await rag_manager.tutor_analyze_mistake(
        target_sentence="I am learning English.",
        user_input="I learning English.",
    )
    call_args = rag_manager._router.generate_explanation.call_args
    prompt = call_args[0][0]
    assert "I am learning English." in prompt
    assert "I learning English." in prompt


def test_build_tutor_prompt_includes_all_parts():
    prompt = _build_tutor_prompt(
        target_sentence="She has gone.",
        user_input="She have went.",
        rag_context="[Context 1] Perfect tense rule.",
    )
    assert "She has gone." in prompt
    assert "She have went." in prompt
    assert "[Context 1]" in prompt


def test_build_tutor_prompt_without_rag_context():
    prompt = _build_tutor_prompt("A.", "B.", "")
    assert "Context" not in prompt


def test_retrieve_respects_top_k(rag_manager):
    for i in range(10):
        rag_manager.add_knowledge(f"Grammar rule number {i}.")
    results = rag_manager.retrieve("grammar rule", top_k=3)
    assert len(results) <= 3


# ── TutorConversation (multi-turn follow-ups) ─────────────────────────────────

def test_start_conversation_returns_tutor_conversation(rag_manager):
    conversation = rag_manager.start_conversation(
        target_sentence="She has gone to the store.",
        user_input="She have went to the store.",
    )
    assert isinstance(conversation, TutorConversation)
    assert len(conversation.messages) == 1
    assert conversation.messages[0]["role"] == "user"


def test_start_conversation_seed_prompt_matches_one_shot(rag_manager):
    """The first-turn prompt should contain the same target/attempt text
    tutor_analyze_mistake() uses, so the two entrypoints stay equivalent."""
    conversation = rag_manager.start_conversation(
        target_sentence="I am learning English.",
        user_input="I learning English.",
    )
    seed_prompt = conversation.messages[0]["content"]
    assert "I am learning English." in seed_prompt
    assert "I learning English." in seed_prompt


async def test_conversation_send_with_no_args_runs_seeded_first_turn(rag_manager):
    conversation = rag_manager.start_conversation("Target.", "Attempt.")
    response = await conversation.send()
    assert response == "Great explanation!"
    assert len(conversation.messages) == 2
    assert conversation.messages[-1]["role"] == "assistant"


async def test_conversation_followup_appends_to_history_not_reset(rag_manager):
    # `messages` is mutated in place after each send(), so a Mock's recorded
    # call_args would just be a live reference showing the FINAL state for
    # every past call. Snapshot (copy) each call's argument at call-time
    # instead, to actually prove what the router saw on each individual call.
    seen_per_call: list[list[dict[str, str]]] = []

    async def fake_chat(messages, system=""):
        seen_per_call.append(list(messages))
        return "Great explanation!"

    rag_manager._router.chat = AsyncMock(side_effect=fake_chat)

    conversation = rag_manager.start_conversation("Target.", "Attempt.")
    await conversation.send()
    await conversation.send("Why is that wrong?")

    assert len(conversation.messages) == 4
    assert conversation.messages[2] == {"role": "user", "content": "Why is that wrong?"}

    # First call: just the seeded prompt (1 turn).
    assert len(seen_per_call[0]) == 1
    # Second call: the ENTIRE history so far (seed + first answer + new
    # question) — that's the whole point of conversation memory, not just
    # the new question in isolation.
    assert len(seen_per_call[1]) == 3
    assert seen_per_call[1][-1]["content"] == "Why is that wrong?"


async def test_conversation_rolls_back_unanswered_turn_on_failure():
    """If the router call fails, any newly-appended user question must be
    rolled back — otherwise the history is left with two consecutive user
    turns and no reply between them, which strict providers (Anthropic)
    reject on the next attempt."""
    store = NumpyVectorStore()
    embedder = HashingEmbeddingProvider()
    failing_router = AsyncMock()
    failing_router.chat = AsyncMock(side_effect=RuntimeError("provider down"))
    rag_manager = RAGManager(store, embedder, failing_router, min_similarity=0.0)

    conversation = rag_manager.start_conversation("Target.", "Attempt.")
    assert len(conversation.messages) == 1  # the seeded first turn

    with pytest.raises(RuntimeError):
        await conversation.send()  # no new text -> nothing to roll back
    assert len(conversation.messages) == 1  # seed survives untouched

    with pytest.raises(RuntimeError):
        await conversation.send("a follow-up that will also fail")
    # The follow-up text must not be left dangling in the history.
    assert len(conversation.messages) == 1
    assert not any(m["content"] == "a follow-up that will also fail" for m in conversation.messages)
