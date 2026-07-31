"""
SM-2 Spaced Repetition Algorithm — Anki-compatible variant.

Reference: Piotr Wozniak's original SM-2 (1987), adapted with Anki's
per-grade ease adjustments and graduation threshold.

Grade semantics:
    1 = Again  — complete failure; card is reset to day-1 relearning
    2 = Hard   — correct but very difficult; shorter-than-normal interval
    3 = Good   — correct with normal effort; standard SM-2 progression
    4 = Easy   — effortless; bonus interval multiplier + ease increase

Ease-factor boundaries:
    - Initial:  2.5
    - Minimum:  1.3  (prevents "ease hell" — cards stuck at tiny intervals)
    - No upper cap (consistent with Anki behaviour)

Graduation:
    A card is considered graduated when its computed interval reaches or
    exceeds GRADUATION_INTERVAL days (default: 21, matching Anki's default).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ── Constants ─────────────────────────────────────────────────────────────────

GRADE_AGAIN = 1
GRADE_HARD = 2
GRADE_GOOD = 3
GRADE_EASY = 4

INITIAL_EASE: float = 2.5
MIN_EASE: float = 1.3
GRADUATION_INTERVAL: int = 21          # days

_EASY_BONUS: float = 1.3              # interval multiplier for Easy answers
_HARD_MULTIPLIER: float = 1.2         # interval multiplier for Hard answers
_EASE_BONUS_EASY: float = 0.15        # ease gain on Easy
_EASE_PENALTY_HARD: float = 0.15      # ease loss on Hard
_EASE_PENALTY_AGAIN: float = 0.20     # ease loss on Again


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReviewResult:
    """
    Immutable snapshot of the new SM-2 state after one review.

    Frozen so it can be passed around safely and compared in tests without
    accidental mutation.
    """

    interval: int           # days until next review
    ease_factor: float
    repetitions: int        # consecutive successful reviews
    next_review_date: datetime
    status: str             # learning | review | graduated


# ── Public API ────────────────────────────────────────────────────────────────

def apply_sm2(
    *,
    interval: int,
    ease_factor: float,
    repetitions: int,
    grade: int,
) -> ReviewResult:
    """
    Apply one SM-2 review step and return the resulting card state.

    Args:
        interval:     Current interval in days.
        ease_factor:  Current ease factor (typically 1.3 – 3.0).
        repetitions:  Number of consecutive successful reviews so far.
        grade:        User quality rating (1=Again, 2=Hard, 3=Good, 4=Easy).

    Returns:
        ReviewResult with the updated SRS state.

    Raises:
        ValueError: If grade is not in {1, 2, 3, 4}.
    """
    if grade not in (GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY):
        raise ValueError(
            f"grade must be 1 (Again), 2 (Hard), 3 (Good), or 4 (Easy) — got {grade!r}"
        )

    new_interval, new_ease, new_reps = _compute_state(
        interval, ease_factor, repetitions, grade
    )

    # Clamp ease to floor after all calculations are done
    new_ease = max(MIN_EASE, new_ease)

    new_status = _determine_status(new_reps, new_interval, grade)
    next_review = datetime.now(timezone.utc) + timedelta(days=new_interval)

    return ReviewResult(
        interval=new_interval,
        ease_factor=round(new_ease, 4),
        repetitions=new_reps,
        next_review_date=next_review,
        status=new_status,
    )


def initial_card_state() -> dict[str, object]:
    """Return the default SM-2 state dictionary for a brand-new card."""
    return {
        "interval": 1,
        "ease_factor": INITIAL_EASE,
        "repetitions": 0,
        "status": "new",
    }


# ── Internal logic ────────────────────────────────────────────────────────────

def _compute_state(
    interval: int,
    ease_factor: float,
    repetitions: int,
    grade: int,
) -> tuple[int, float, int]:
    """Return (new_interval, new_ease, new_reps) — before ease clamping."""

    if grade == GRADE_AGAIN:
        return 1, ease_factor - _EASE_PENALTY_AGAIN, 0

    new_reps = repetitions + 1

    if grade == GRADE_HARD:
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(interval * _HARD_MULTIPLIER))
        return new_interval, ease_factor - _EASE_PENALTY_HARD, new_reps

    if grade == GRADE_GOOD:
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = round(interval * ease_factor)
        return new_interval, ease_factor, new_reps

    # GRADE_EASY
    if repetitions == 0:
        new_interval = 4
    elif repetitions == 1:
        new_interval = 6
    else:
        new_interval = round(interval * ease_factor * _EASY_BONUS)
    return new_interval, ease_factor + _EASE_BONUS_EASY, new_reps


def _determine_status(reps: int, interval: int, grade: int) -> str:
    """Map (reps, interval, grade) to a CardStatus string."""
    if grade == GRADE_AGAIN:
        return "learning"
    if interval >= GRADUATION_INTERVAL:
        return "graduated"
    if reps >= 2:
        return "review"
    return "learning"
