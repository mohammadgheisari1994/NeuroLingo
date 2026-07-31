"""
Shared pytest fixtures for unit tests.

All DB fixtures use tmp_path (a real SQLite file in a pytest-managed temp
directory) so production and test code exercise exactly the same code path.
The sqlite3 in-memory mode is intentionally avoided because each new
connection() call to ':memory:' creates a separate database, which would
break the context-manager-per-query pattern used by DatabaseRepository.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from logger_config import setup_logging
from neurolingo.db.models import Card, Sentence
from neurolingo.db.repository import DatabaseRepository


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Set up logging once for the entire test session at DEBUG level."""
    setup_logging(level="DEBUG")


@pytest.fixture
def repo(tmp_path) -> DatabaseRepository:
    """Fresh DatabaseRepository backed by a per-test SQLite file."""
    r = DatabaseRepository(tmp_path / "test.db")
    r.create_schema()
    return r


@pytest.fixture
def sentence(repo) -> Sentence:
    """A pre-inserted Sentence for tests that need an existing sentence."""
    return repo.add_sentence(Sentence(
        sentence_en="The quick brown fox jumps over the lazy dog.",
        sentence_fa="روباه قهوه‌ای چابک از روی سگ تنبل می‌پرد.",
        context_notes="Classic pangram used for font samples.",
        source="test_fixture",
    ))


@pytest.fixture
def card(repo, sentence) -> Card:
    """A pre-inserted Card in 'review' status with a past due date."""
    return repo.add_card(Card(
        sentence_id=sentence.id,
        interval=6,
        ease_factor=2.5,
        repetitions=2,
        next_review_date=datetime.now(timezone.utc) - timedelta(days=1),
        status="review",
    ))
