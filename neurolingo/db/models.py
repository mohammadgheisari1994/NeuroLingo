"""
Domain dataclasses for NeuroLingo.

Plain Python dataclasses act as typed DTOs between the repository layer and
the rest of the application.  No ORM dependency — sqlite3 is stdlib, zero
extra mobile footprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class CardStatus(str, Enum):
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    GRADUATED = "graduated"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Sentence:
    """
    A full English sentence paired with its Farsi translation.

    This is the atomic unit of Hebbian chunking: vocabulary is never stored
    or reviewed in isolation — always inside a complete sentence.
    """

    sentence_en: str
    sentence_fa: str
    context_notes: str = ""
    source: str = ""          # e.g. "podcast_ep12", "user_import"
    id: int | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)


@dataclass
class Card:
    """
    SM-2 flashcard state linked to exactly one Sentence.

    All SRS fields are owned here; the algorithm module reads them, computes
    the next state, and the repository persists the result.
    """

    sentence_id: int
    interval: int = 1              # days until next review
    ease_factor: float = 2.5       # SM-2 multiplier; floor is 1.3
    repetitions: int = 0           # consecutive successful reviews
    next_review_date: datetime = field(default_factory=_now_utc)
    status: str = CardStatus.NEW.value   # new | learning | review | graduated
    id: int | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)


@dataclass
class ReviewLog:
    """
    Append-only record of every review action.

    Never mutated after creation — the accumulation of these rows is the
    source of truth for the user's personal learning analytics.
    """

    card_id: int
    grade: int            # 1=Again  2=Hard  3=Good  4=Easy
    old_interval: int
    new_interval: int
    old_ease_factor: float
    new_ease_factor: float
    id: int | None = None
    reviewed_at: datetime = field(default_factory=_now_utc)
