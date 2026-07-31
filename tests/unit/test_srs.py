"""
Unit tests for neurolingo.core.srs.algorithm (SM-2 implementation).

Every constant and formula branch is tested explicitly with hand-calculated
expected values so regressions are caught immediately.

Grade semantics (repeated here for readability):
    1 = Again   2 = Hard   3 = Good   4 = Easy
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from neurolingo.core.srs.algorithm import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    GRADUATION_INTERVAL,
    INITIAL_EASE,
    MIN_EASE,
    ReviewResult,
    apply_sm2,
    initial_card_state,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _review(
    *,
    reps: int = 0,
    interval: int = 1,
    ease: float = INITIAL_EASE,
    grade: int,
) -> ReviewResult:
    """Thin wrapper so tests read like specifications."""
    return apply_sm2(
        interval=interval,
        ease_factor=ease,
        repetitions=reps,
        grade=grade,
    )


# ── Input validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_grade", [0, 5, -1, 99])
def test_invalid_grade_raises_value_error(bad_grade):
    with pytest.raises(ValueError, match="grade must be"):
        _review(grade=bad_grade)


# ── Again (grade=1) ───────────────────────────────────────────────────────────

def test_again_first_review():
    r = _review(reps=0, grade=GRADE_AGAIN)
    assert r.interval == 1
    assert r.repetitions == 0
    assert r.status == "learning"


def test_again_resets_mature_card():
    """A card with 100-day interval must be reset completely."""
    r = _review(reps=10, interval=100, ease=2.5, grade=GRADE_AGAIN)
    assert r.interval == 1
    assert r.repetitions == 0
    assert r.ease_factor == pytest.approx(2.3)
    assert r.status == "learning"


def test_again_decreases_ease_by_020():
    r = _review(ease=2.5, grade=GRADE_AGAIN)
    assert r.ease_factor == pytest.approx(2.3)


def test_again_does_not_drop_ease_below_floor():
    """ease_factor must never fall below MIN_EASE (1.3)."""
    r = _review(ease=MIN_EASE + 0.05, grade=GRADE_AGAIN)
    assert r.ease_factor == pytest.approx(MIN_EASE)


def test_again_at_ease_floor_stays_at_floor():
    r = _review(ease=MIN_EASE, grade=GRADE_AGAIN)
    assert r.ease_factor == pytest.approx(MIN_EASE)


# ── Hard (grade=2) ────────────────────────────────────────────────────────────

def test_hard_first_review_interval_is_1():
    r = _review(reps=0, grade=GRADE_HARD)
    assert r.interval == 1
    assert r.repetitions == 1
    assert r.ease_factor == pytest.approx(2.35)   # 2.5 - 0.15
    assert r.status == "learning"


def test_hard_second_review_interval_is_6():
    r = _review(reps=1, interval=1, grade=GRADE_HARD)
    assert r.interval == 6


def test_hard_mature_uses_12x_multiplier():
    # reps=2, interval=10 → round(10 * 1.2) = 12
    r = _review(reps=2, interval=10, ease=2.5, grade=GRADE_HARD)
    assert r.interval == 12
    assert r.ease_factor == pytest.approx(2.35)


def test_hard_decreases_ease_by_015():
    r = _review(reps=2, interval=6, ease=2.5, grade=GRADE_HARD)
    assert r.ease_factor == pytest.approx(2.35)


def test_hard_ease_cannot_go_below_floor():
    r = _review(reps=2, interval=6, ease=1.35, grade=GRADE_HARD)
    assert r.ease_factor == pytest.approx(MIN_EASE)


def test_hard_status_is_review_after_two_reps():
    r = _review(reps=1, interval=1, ease=2.5, grade=GRADE_HARD)
    assert r.status == "review"


# ── Good (grade=3) ────────────────────────────────────────────────────────────

def test_good_first_review_interval_is_1():
    r = _review(reps=0, grade=GRADE_GOOD)
    assert r.interval == 1
    assert r.repetitions == 1
    assert r.status == "learning"


def test_good_second_review_interval_is_6():
    r = _review(reps=1, interval=1, grade=GRADE_GOOD)
    assert r.interval == 6
    assert r.repetitions == 2
    assert r.status == "review"


def test_good_third_review_uses_ease_multiplier():
    # reps=2, interval=6, ease=2.5 → round(6 * 2.5) = 15
    r = _review(reps=2, interval=6, ease=2.5, grade=GRADE_GOOD)
    assert r.interval == 15
    assert r.repetitions == 3
    assert r.ease_factor == pytest.approx(2.5)    # unchanged for Good
    assert r.status == "review"                   # 15 < GRADUATION_INTERVAL


def test_good_does_not_change_ease():
    r = _review(reps=2, interval=6, ease=1.8, grade=GRADE_GOOD)
    assert r.ease_factor == pytest.approx(1.8)


def test_good_graduates_when_interval_reaches_threshold():
    # reps=3, interval=10, ease=2.5 → round(10 * 2.5) = 25 ≥ 21
    r = _review(reps=3, interval=10, ease=2.5, grade=GRADE_GOOD)
    assert r.interval == 25
    assert r.status == "graduated"


def test_good_stays_review_just_below_graduation():
    # reps=3, interval=8, ease=2.5 → round(8 * 2.5) = 20 < 21
    r = _review(reps=3, interval=8, ease=2.5, grade=GRADE_GOOD)
    assert r.interval == 20
    assert r.status == "review"


# ── Easy (grade=4) ────────────────────────────────────────────────────────────

def test_easy_first_review_interval_is_4():
    r = _review(reps=0, grade=GRADE_EASY)
    assert r.interval == 4
    assert r.repetitions == 1
    assert r.ease_factor == pytest.approx(2.65)   # 2.5 + 0.15
    assert r.status == "learning"                  # 4 < 21


def test_easy_second_review_interval_is_6():
    r = _review(reps=1, interval=4, grade=GRADE_EASY)
    assert r.interval == 6


def test_easy_mature_uses_ease_and_bonus():
    # reps=3, interval=10, ease=2.5 → round(10 * 2.5 * 1.3) = round(32.5) = 32
    r = _review(reps=3, interval=10, ease=2.5, grade=GRADE_EASY)
    assert r.interval == 32
    assert r.ease_factor == pytest.approx(2.65)
    assert r.status == "graduated"                 # 32 ≥ 21


def test_easy_increases_ease_by_015():
    r = _review(reps=2, interval=6, ease=2.5, grade=GRADE_EASY)
    assert r.ease_factor == pytest.approx(2.65)


def test_easy_ease_increase_is_cumulative():
    r1 = _review(reps=2, interval=6, ease=1.5, grade=GRADE_EASY)
    assert r1.ease_factor == pytest.approx(1.65)


# ── Status transitions ────────────────────────────────────────────────────────

@pytest.mark.parametrize("grade", [GRADE_GOOD, GRADE_EASY, GRADE_HARD])
def test_first_review_status_is_learning(grade):
    r = _review(reps=0, grade=grade)
    assert r.status == "learning"


def test_status_becomes_review_at_two_reps():
    r = _review(reps=1, interval=1, grade=GRADE_GOOD)
    assert r.repetitions == 2
    assert r.status == "review"


def test_again_on_graduated_card_resets_to_learning():
    r = _review(reps=20, interval=180, ease=2.5, grade=GRADE_AGAIN)
    assert r.status == "learning"
    assert r.interval == 1


def test_graduation_interval_constant_is_correct():
    """Validate the graduation boundary against a known good/bad pair."""
    below = _review(reps=3, interval=8, ease=2.5, grade=GRADE_GOOD)   # → 20
    above = _review(reps=3, interval=10, ease=2.5, grade=GRADE_GOOD)  # → 25
    assert below.status == "review"
    assert above.status == "graduated"
    assert GRADUATION_INTERVAL == 21


# ── next_review_date ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("grade", [GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY])
def test_next_review_date_is_always_in_the_future(grade):
    r = _review(reps=2, interval=6, grade=grade)
    assert r.next_review_date > datetime.now(timezone.utc)


def test_next_review_date_reflects_interval():
    from datetime import timedelta
    r = _review(reps=2, interval=6, ease=2.5, grade=GRADE_GOOD)
    # interval=15 → review in ≈15 days
    expected = datetime.now(timezone.utc) + timedelta(days=r.interval)
    delta = abs((r.next_review_date - expected).total_seconds())
    assert delta < 5, "next_review_date should be within 5 s of now + interval"


# ── Immutability ──────────────────────────────────────────────────────────────

def test_review_result_is_frozen():
    r = _review(reps=0, grade=GRADE_GOOD)
    with pytest.raises((TypeError, AttributeError)):
        r.interval = 999  # type: ignore[misc]


# ── initial_card_state ────────────────────────────────────────────────────────

def test_initial_card_state_defaults():
    state = initial_card_state()
    assert state["interval"] == 1
    assert state["ease_factor"] == INITIAL_EASE
    assert state["repetitions"] == 0
    assert state["status"] == "new"


def test_initial_ease_constant():
    assert INITIAL_EASE == 2.5


def test_min_ease_constant():
    assert MIN_EASE == 1.3
