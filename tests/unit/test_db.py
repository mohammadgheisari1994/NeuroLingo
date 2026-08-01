"""
Unit tests for neurolingo.db.repository (DatabaseRepository).

Coverage targets:
- Schema creation
- Sentence CRUD
- Card CRUD + state persistence
- FK constraint enforcement
- ON DELETE CASCADE across all three tables
- get_due_cards / get_new_cards filtering logic
- ReviewLog append and history ordering
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from neurolingo.db.models import Card, CardStatus, ReviewLog, Sentence

# ── Schema ────────────────────────────────────────────────────────────────────

def test_create_schema_is_idempotent(repo):
    """Calling create_schema() twice must not raise."""
    repo.create_schema()  # second call — should be a no-op


# ── Sentences ─────────────────────────────────────────────────────────────────

def test_add_sentence_assigns_integer_id(repo):
    s = repo.add_sentence(Sentence(
        sentence_en="Practice makes perfect.",
        sentence_fa="تمرین موجب کمال می‌شود.",
    ))
    assert isinstance(s.id, int)
    assert s.id > 0


def test_add_and_get_sentence_roundtrip(repo):
    original = Sentence(
        sentence_en="Every cloud has a silver lining.",
        sentence_fa="هر ابری نقره‌ای دارد.",
        context_notes="Idiom: optimism",
        source="proverb_pack",
    )
    saved = repo.add_sentence(original)
    fetched = repo.get_sentence(saved.id)

    assert fetched is not None
    assert fetched.sentence_en == original.sentence_en
    assert fetched.sentence_fa == original.sentence_fa
    assert fetched.context_notes == original.context_notes
    assert fetched.source == original.source


def test_get_sentence_returns_none_for_missing_id(repo):
    assert repo.get_sentence(99999) is None


def test_count_sentences_reflects_inserts(repo):
    assert repo.count_sentences() == 0
    repo.add_sentence(Sentence(sentence_en="One.", sentence_fa="یک."))
    repo.add_sentence(Sentence(sentence_en="Two.", sentence_fa="دو."))
    assert repo.count_sentences() == 2


def test_count_sentences_reflects_deletes(repo):
    s = repo.add_sentence(Sentence(sentence_en="Gone soon.", sentence_fa="به زودی رفت."))
    assert repo.count_sentences() == 1
    repo.delete_sentence(s.id)
    assert repo.count_sentences() == 0


def test_delete_sentence_removes_row(repo, sentence):
    repo.delete_sentence(sentence.id)
    assert repo.get_sentence(sentence.id) is None


# ── Cards ─────────────────────────────────────────────────────────────────────

def test_add_card_assigns_integer_id(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))
    assert isinstance(c.id, int)
    assert c.id > 0


def test_add_card_rejects_nonexistent_sentence(repo):
    """FK constraint must prevent orphaned cards."""
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_card(Card(sentence_id=99999))


def test_get_card_roundtrip(repo, sentence):
    added = repo.add_card(Card(
        sentence_id=sentence.id,
        interval=10,
        ease_factor=2.3,
        repetitions=3,
        status=CardStatus.REVIEW.value,
    ))
    fetched = repo.get_card(added.id)

    assert fetched is not None
    assert fetched.interval == 10
    assert fetched.ease_factor == pytest.approx(2.3)
    assert fetched.repetitions == 3
    assert fetched.status == CardStatus.REVIEW.value


def test_get_card_returns_none_for_missing_id(repo):
    assert repo.get_card(99999) is None


def test_update_card_persists_all_srs_fields(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))

    c.interval = 21
    c.ease_factor = 1.8
    c.repetitions = 5
    c.status = CardStatus.GRADUATED.value
    repo.update_card(c)

    fetched = repo.get_card(c.id)
    assert fetched.interval == 21
    assert fetched.ease_factor == pytest.approx(1.8)
    assert fetched.repetitions == 5
    assert fetched.status == CardStatus.GRADUATED.value


def test_update_card_refreshes_updated_at(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))
    original_updated_at = c.updated_at

    c.interval = 7
    repo.update_card(c)

    fetched = repo.get_card(c.id)
    assert fetched.updated_at >= original_updated_at


# ── Cascade deletes ───────────────────────────────────────────────────────────

def test_delete_sentence_cascades_to_card(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))
    repo.delete_sentence(sentence.id)
    assert repo.get_card(c.id) is None


def test_delete_sentence_cascades_to_review_log(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))
    repo.add_review_log(ReviewLog(
        card_id=c.id, grade=3,
        old_interval=1, new_interval=6,
        old_ease_factor=2.5, new_ease_factor=2.5,
    ))

    repo.delete_sentence(sentence.id)
    assert repo.get_review_history(c.id) == []


# ── get_due_cards ─────────────────────────────────────────────────────────────

def test_get_due_cards_includes_overdue(repo, sentence):
    overdue = repo.add_card(Card(
        sentence_id=sentence.id,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=3),
        status=CardStatus.REVIEW.value,
    ))
    due = repo.get_due_cards()
    assert any(d.id == overdue.id for d in due)


def test_get_due_cards_excludes_future(repo, sentence):
    repo.add_card(Card(
        sentence_id=sentence.id,
        next_review_date=datetime.now(timezone.utc) + timedelta(days=5),
        status=CardStatus.REVIEW.value,
    ))
    assert repo.get_due_cards() == []


def test_get_due_cards_excludes_new_status(repo, sentence):
    """New cards should not appear in the due queue."""
    repo.add_card(Card(
        sentence_id=sentence.id,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=1),
        status=CardStatus.NEW.value,   # new — must be excluded
    ))
    assert repo.get_due_cards() == []


def test_get_due_cards_honours_limit(repo):
    """Results must be capped at the requested limit."""
    for i in range(5):
        s = repo.add_sentence(Sentence(f"Sentence {i}", f"جمله {i}"))
        repo.add_card(Card(
            sentence_id=s.id,
            next_review_date=datetime.now(timezone.utc) - timedelta(days=i + 1),
            status=CardStatus.REVIEW.value,
        ))
    assert len(repo.get_due_cards(limit=3)) == 3


def test_get_due_cards_ordered_most_overdue_first(repo):
    s1 = repo.add_sentence(Sentence("A", "الف"))
    s2 = repo.add_sentence(Sentence("B", "ب"))
    old_card = repo.add_card(Card(
        sentence_id=s1.id,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=10),
        status=CardStatus.REVIEW.value,
    ))
    new_card = repo.add_card(Card(
        sentence_id=s2.id,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=1),
        status=CardStatus.REVIEW.value,
    ))
    due = repo.get_due_cards()
    assert due[0].id == old_card.id
    assert due[1].id == new_card.id


# ── get_new_cards ─────────────────────────────────────────────────────────────

def test_get_new_cards_returns_only_new_status(repo, sentence):
    new_card = repo.add_card(Card(
        sentence_id=sentence.id,
        status=CardStatus.NEW.value,
    ))
    repo.add_card(Card(
        sentence_id=sentence.id,
        status=CardStatus.REVIEW.value,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=1),
    ))
    new_cards = repo.get_new_cards()
    ids = [c.id for c in new_cards]
    assert new_card.id in ids
    assert all(c.status == CardStatus.NEW.value for c in new_cards)


# ── count_cards_by_status ──────────────────────────────────────────────────────

def test_count_cards_by_status_zero_when_none_match(repo):
    assert repo.count_cards_by_status("graduated") == 0


def test_count_cards_by_status_counts_only_matching_status(repo, sentence):
    repo.add_card(Card(sentence_id=sentence.id, status=CardStatus.GRADUATED.value))
    repo.add_card(Card(sentence_id=sentence.id, status=CardStatus.GRADUATED.value))
    repo.add_card(Card(sentence_id=sentence.id, status=CardStatus.NEW.value))
    assert repo.count_cards_by_status("graduated") == 2
    assert repo.count_cards_by_status("new") == 1


# ── search_sentences ────────────────────────────────────────────────────────────

def test_search_sentences_empty_query_returns_everything(repo):
    repo.add_sentence(Sentence(sentence_en="One.", sentence_fa="یک."))
    repo.add_sentence(Sentence(sentence_en="Two.", sentence_fa="دو."))
    assert len(repo.search_sentences("")) == 2
    assert len(repo.search_sentences("   ")) == 2


def test_search_sentences_matches_english_case_insensitively(repo):
    repo.add_sentence(Sentence(sentence_en="Despite the heavy rain.", sentence_fa="X"))
    repo.add_sentence(Sentence(sentence_en="A sunny day.", sentence_fa="Y"))
    results = repo.search_sentences("RAIN")
    assert len(results) == 1
    assert "rain" in results[0].sentence_en.lower()


def test_search_sentences_matches_farsi_text(repo):
    repo.add_sentence(Sentence(sentence_en="It rained.", sentence_fa="باران آمد."))
    repo.add_sentence(Sentence(sentence_en="It was sunny.", sentence_fa="آفتابی بود."))
    results = repo.search_sentences("باران")
    assert len(results) == 1


def test_search_sentences_matches_context_notes(repo):
    repo.add_sentence(Sentence(
        sentence_en="If I were you...", sentence_fa="X",
        context_notes="Second conditional",
    ))
    repo.add_sentence(Sentence(sentence_en="Other.", sentence_fa="Y"))
    results = repo.search_sentences("conditional")
    assert len(results) == 1


def test_search_sentences_no_match_returns_empty(repo, sentence):
    assert repo.search_sentences("no such word anywhere") == []


# ── ReviewLog ─────────────────────────────────────────────────────────────────

def test_add_review_log_assigns_id(repo, card):
    entry = repo.add_review_log(ReviewLog(
        card_id=card.id, grade=3,
        old_interval=1, new_interval=6,
        old_ease_factor=2.5, new_ease_factor=2.5,
    ))
    assert isinstance(entry.id, int)
    assert entry.id > 0


def test_get_review_history_returns_all_entries(repo, card):
    for grade in [3, 4, 2]:
        repo.add_review_log(ReviewLog(
            card_id=card.id, grade=grade,
            old_interval=1, new_interval=grade * 2,
            old_ease_factor=2.5, new_ease_factor=2.5,
        ))
    history = repo.get_review_history(card.id)
    assert len(history) == 3
    assert [h.grade for h in history] == [3, 4, 2]


def test_review_history_is_insertion_ordered(repo, card):
    """History must be ordered by insertion id, not by timestamp."""
    grades = [1, 3, 4, 2, 3]
    for g in grades:
        repo.add_review_log(ReviewLog(
            card_id=card.id, grade=g,
            old_interval=1, new_interval=1,
            old_ease_factor=2.5, new_ease_factor=2.5,
        ))
    history = repo.get_review_history(card.id)
    assert [h.grade for h in history] == grades


def test_review_history_empty_for_unreviewed_card(repo, sentence):
    c = repo.add_card(Card(sentence_id=sentence.id))
    assert repo.get_review_history(c.id) == []


def test_review_log_rejects_nonexistent_card(repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_review_log(ReviewLog(
            card_id=99999, grade=3,
            old_interval=1, new_interval=6,
            old_ease_factor=2.5, new_ease_factor=2.5,
        ))
