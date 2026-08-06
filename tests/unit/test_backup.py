"""
Unit tests for neurolingo.db.backup (full data export/import).

Coverage targets:
- export_backup() serialises sentences + cards + review history correctly
- import_backup() round-trips that data into a fresh repository
- Malformed / unsupported-version input is rejected with BackupFormatError
- Repeated import duplicates rather than merges (documented behaviour)
"""
from __future__ import annotations

import pytest

from neurolingo.core.llm.router import LLMRouter
from neurolingo.core.rag.embeddings import HashingEmbeddingProvider
from neurolingo.core.rag.rag_manager import RAGManager
from neurolingo.core.rag.vectorstore import NumpyVectorStore
from neurolingo.db.backup import (
    BACKUP_FORMAT_VERSION,
    BackupFormatError,
    export_backup,
    import_backup,
)
from neurolingo.db.models import ReviewLog, Sentence
from neurolingo.db.repository import DatabaseRepository


def _build_rag(min_similarity: float = 0.0) -> tuple[RAGManager, NumpyVectorStore]:
    store = NumpyVectorStore(persist_path=None)
    router = LLMRouter(providers=[], local_provider=None)
    rag = RAGManager(store, HashingEmbeddingProvider(), router, min_similarity=min_similarity)
    return rag, store

# ── get_all_sentences / get_card_by_sentence_id ───────────────────────────────

def test_get_all_sentences_empty(repo):
    assert repo.get_all_sentences() == []


def test_get_all_sentences_returns_everything_in_order(repo):
    repo.add_sentence(Sentence(sentence_en="One.", sentence_fa="یک."))
    repo.add_sentence(Sentence(sentence_en="Two.", sentence_fa="دو."))
    sentences = repo.get_all_sentences()
    assert [s.sentence_en for s in sentences] == ["One.", "Two."]


def test_get_card_by_sentence_id_returns_none_when_no_card(repo, sentence):
    assert repo.get_card_by_sentence_id(sentence.id) is None


def test_get_card_by_sentence_id_finds_the_card(repo, sentence, card):
    found = repo.get_card_by_sentence_id(sentence.id)
    assert found is not None
    assert found.id == card.id


# ── export_backup() ────────────────────────────────────────────────────────────

def test_export_empty_repo(repo):
    backup = export_backup(repo)
    assert backup["format_version"] == BACKUP_FORMAT_VERSION
    assert backup["sentences"] == []
    assert "exported_at" in backup


def test_export_sentence_with_no_card_yet(repo, sentence):
    backup = export_backup(repo)
    assert len(backup["sentences"]) == 1
    entry = backup["sentences"][0]
    assert entry["sentence_en"] == sentence.sentence_en
    assert entry["card"] is None
    assert entry["review_log"] == []


def test_export_includes_card_state(repo, sentence, card):
    backup = export_backup(repo)
    entry = backup["sentences"][0]
    assert entry["card"]["interval"] == card.interval
    assert entry["card"]["status"] == card.status


def test_export_includes_review_history(repo, sentence, card):
    repo.add_review_log(ReviewLog(
        card_id=card.id, grade=3,
        old_interval=1, new_interval=6,
        old_ease_factor=2.5, new_ease_factor=2.5,
    ))
    backup = export_backup(repo)
    entry = backup["sentences"][0]
    assert len(entry["review_log"]) == 1
    assert entry["review_log"][0]["grade"] == 3


# ── import_backup() ────────────────────────────────────────────────────────────

def test_import_rejects_missing_sentences_key(repo):
    with pytest.raises(BackupFormatError, match="sentences"):
        import_backup(repo, {"format_version": BACKUP_FORMAT_VERSION})


def test_import_rejects_non_dict(repo):
    with pytest.raises(BackupFormatError):
        import_backup(repo, ["not", "a", "dict"])


def test_import_rejects_unsupported_version(repo):
    with pytest.raises(BackupFormatError, match="format_version"):
        import_backup(repo, {"format_version": 999, "sentences": []})


def test_export_then_import_roundtrips_into_fresh_repo(repo, sentence, card, tmp_path):
    repo.add_review_log(ReviewLog(
        card_id=card.id, grade=4,
        old_interval=1, new_interval=4,
        old_ease_factor=2.5, new_ease_factor=2.65,
    ))
    backup = export_backup(repo)

    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    imported_count = import_backup(fresh, backup)

    assert imported_count == 1
    restored_sentences = fresh.get_all_sentences()
    assert len(restored_sentences) == 1
    assert restored_sentences[0].sentence_en == sentence.sentence_en

    restored_card = fresh.get_card_by_sentence_id(restored_sentences[0].id)
    assert restored_card is not None
    assert restored_card.interval == card.interval
    assert restored_card.status == card.status

    history = fresh.get_review_history(restored_card.id)
    assert len(history) == 1
    assert history[0].grade == 4


def test_import_sentence_with_no_card_creates_no_card(repo, sentence, tmp_path):
    backup = export_backup(repo)
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    import_backup(fresh, backup)

    restored = fresh.get_all_sentences()[0]
    assert fresh.get_card_by_sentence_id(restored.id) is None


def test_reimporting_the_same_backup_duplicates_not_merges(repo, sentence, tmp_path):
    backup = export_backup(repo)
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()

    import_backup(fresh, backup)
    import_backup(fresh, backup)

    assert fresh.count_sentences() == 2  # documented behaviour, not a bug


# ── import_backup() RAG indexing (#48) ─────────────────────────────────────────

def test_import_without_rag_does_not_touch_knowledge_base(repo, sentence, tmp_path):
    """Passing no `rag` (the pre-fix default) must keep working exactly as
    before — importing shouldn't require a RAGManager."""
    backup = export_backup(repo)
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    imported = import_backup(fresh, backup)
    assert imported == 1


def test_import_with_rag_indexes_each_sentence(repo, sentence, tmp_path):
    backup = export_backup(repo)
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    rag, store = _build_rag()

    import_backup(fresh, backup, rag=rag)

    assert len(store) == 1
    results = rag.retrieve(sentence.sentence_en, top_k=1)
    assert results
    assert sentence.sentence_en in results[0]["text"]


def test_import_with_rag_indexes_every_sentence_in_a_multi_entry_backup(tmp_path):
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    rag, store = _build_rag()
    backup = {
        "format_version": BACKUP_FORMAT_VERSION,
        "sentences": [
            {"sentence_en": "One.", "sentence_fa": "یک."},
            {"sentence_en": "Two.", "sentence_fa": "دو.", "context_notes": "counting"},
        ],
    }

    import_backup(fresh, backup, rag=rag)

    assert len(store) == 2


def test_import_new_style_sentence_missing_optional_fields(tmp_path):
    """context_notes/source are optional on the way in — a hand-edited or
    older-format backup shouldn't crash on missing keys."""
    fresh = DatabaseRepository(tmp_path / "fresh.db")
    fresh.create_schema()
    minimal_backup = {
        "format_version": BACKUP_FORMAT_VERSION,
        "sentences": [{"sentence_en": "Hi.", "sentence_fa": "سلام."}],
    }
    count = import_backup(fresh, minimal_backup)
    assert count == 1
    assert fresh.get_all_sentences()[0].context_notes == ""
