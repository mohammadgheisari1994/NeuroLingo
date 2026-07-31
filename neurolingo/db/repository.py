"""
DatabaseRepository: all SQLite I/O for NeuroLingo.

Design principles:
- Every public method is fully self-contained; it opens, uses, and closes its
  own connection via the _connect() context manager so there is never a
  dangling file lock.
- Foreign keys are enforced on every connection via PRAGMA.
- All exceptions are caught, logged with full stack traces, and re-raised so
  callers can decide on recovery strategy.
- WAL journal mode is requested for better concurrent read performance; SQLite
  silently ignores this for :memory: databases, so tests are unaffected.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from logger_config import get_logger
from neurolingo.db.models import Card, ReviewLog, Sentence

_log = get_logger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_en   TEXT    NOT NULL,
    sentence_fa   TEXT    NOT NULL,
    context_notes TEXT    NOT NULL DEFAULT '',
    source        TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id      INTEGER NOT NULL
                         REFERENCES sentences(id) ON DELETE CASCADE,
    interval         INTEGER NOT NULL DEFAULT 1,
    ease_factor      REAL    NOT NULL DEFAULT 2.5,
    repetitions      INTEGER NOT NULL DEFAULT 0,
    next_review_date TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'new',
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS review_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id         INTEGER NOT NULL
                        REFERENCES cards(id) ON DELETE CASCADE,
    grade           INTEGER NOT NULL,
    old_interval    INTEGER NOT NULL,
    new_interval    INTEGER NOT NULL,
    old_ease_factor REAL    NOT NULL,
    new_ease_factor REAL    NOT NULL,
    reviewed_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cards_next_review
    ON cards (next_review_date, status);

CREATE INDEX IF NOT EXISTS idx_review_log_card
    ON review_log (card_id, reviewed_at);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    """Serialise an aware datetime to an ISO-8601 string for SQLite storage."""
    return dt.isoformat()


def _parse(value: str | None) -> datetime:
    """Parse an ISO-8601 string back to an aware datetime."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value)
        # Ensure the result is always timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        _log.warning("Could not parse datetime string %r; using now(UTC)", value)
        return datetime.now(timezone.utc)


# ── Repository ────────────────────────────────────────────────────────────────

class DatabaseRepository:
    """
    Thread-local SQLite repository.

    Instantiate once per application (or test), call create_schema() on
    first use, then use the CRUD methods.  Each method manages its own
    connection lifetime; there is no shared connection state.
    """

    def __init__(self, db_path: str | Path = "data/neurolingo.db") -> None:
        self._db_path = str(db_path)
        _log.info("DatabaseRepository initialised | path=%s", self._db_path)

    # ── Connection ────────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Open a connection, yield it, commit on clean exit, rollback on error.

        Foreign keys and WAL mode are enabled on every new connection so
        behaviour is consistent regardless of which method is called first.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            _log.exception(
                "Transaction rolled back | db=%s", self._db_path
            )
            raise
        finally:
            conn.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """
        Create all tables and indexes if they do not already exist.

        Safe to call on every application startup — all statements use
        CREATE IF NOT EXISTS.  Uses executescript() because it handles
        multi-statement DDL; the implicit COMMIT it issues is harmless.
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA_SQL)
            conn.close()
            _log.info("Schema verified/created | db=%s", self._db_path)
        except Exception:
            _log.exception("Schema creation failed | db=%s", self._db_path)
            raise

    # ── Sentences ─────────────────────────────────────────────────────────────

    def add_sentence(self, sentence: Sentence) -> Sentence:
        """Insert a new sentence row and return it with its generated id."""
        sql = """
        INSERT INTO sentences
            (sentence_en, sentence_fa, context_notes, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, (
                    sentence.sentence_en,
                    sentence.sentence_fa,
                    sentence.context_notes,
                    sentence.source,
                    _fmt(sentence.created_at),
                    _fmt(sentence.updated_at),
                ))
                sentence.id = cur.lastrowid
            _log.debug(
                "Sentence added | id=%d | en=%.50s",
                sentence.id, sentence.sentence_en,
            )
            return sentence
        except Exception:
            _log.exception("add_sentence failed")
            raise

    def get_sentence(self, sentence_id: int) -> Sentence | None:
        """Return the Sentence with the given id, or None if not found."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sentences WHERE id = ?", (sentence_id,)
                ).fetchone()
            return _row_to_sentence(row) if row else None
        except Exception:
            _log.exception("get_sentence failed | id=%d", sentence_id)
            raise

    def count_sentences(self) -> int:
        """Return the total number of sentences stored, regardless of card status."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM sentences").fetchone()
            return int(row["n"])
        except Exception:
            _log.exception("count_sentences failed")
            raise

    def get_all_sentences(self) -> list[Sentence]:
        """Return every sentence, oldest first — used for full data export."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM sentences ORDER BY id ASC").fetchall()
            return [_row_to_sentence(r) for r in rows]
        except Exception:
            _log.exception("get_all_sentences failed")
            raise

    def delete_sentence(self, sentence_id: int) -> None:
        """
        Delete a sentence.  Cards and review_log entries are removed
        automatically via ON DELETE CASCADE.
        """
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM sentences WHERE id = ?", (sentence_id,))
            _log.debug("Sentence deleted | id=%d (cascade applied)", sentence_id)
        except Exception:
            _log.exception("delete_sentence failed | id=%d", sentence_id)
            raise

    # ── Cards ─────────────────────────────────────────────────────────────────

    def add_card(self, card: Card) -> Card:
        """Insert a new card and return it with its generated id."""
        sql = """
        INSERT INTO cards
            (sentence_id, interval, ease_factor, repetitions,
             next_review_date, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, (
                    card.sentence_id,
                    card.interval,
                    card.ease_factor,
                    card.repetitions,
                    _fmt(card.next_review_date),
                    card.status,
                    _fmt(card.created_at),
                    _fmt(card.updated_at),
                ))
                card.id = cur.lastrowid
            _log.debug(
                "Card added | id=%d | sentence_id=%d | status=%s",
                card.id, card.sentence_id, card.status,
            )
            return card
        except Exception:
            _log.exception("add_card failed | sentence_id=%d", card.sentence_id)
            raise

    def get_card(self, card_id: int) -> Card | None:
        """Return the Card with the given id, or None if not found."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM cards WHERE id = ?", (card_id,)
                ).fetchone()
            return _row_to_card(row) if row else None
        except Exception:
            _log.exception("get_card failed | id=%d", card_id)
            raise

    def get_card_by_sentence_id(self, sentence_id: int) -> Card | None:
        """Return the card linked to a sentence, or None if it has none yet
        — used for full data export, where we walk sentences first."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM cards WHERE sentence_id = ? LIMIT 1", (sentence_id,)
                ).fetchone()
            return _row_to_card(row) if row else None
        except Exception:
            _log.exception("get_card_by_sentence_id failed | sentence_id=%d", sentence_id)
            raise

    def update_card(self, card: Card) -> None:
        """Persist the updated SM-2 state for an existing card."""
        sql = """
        UPDATE cards SET
            interval         = ?,
            ease_factor      = ?,
            repetitions      = ?,
            next_review_date = ?,
            status           = ?,
            updated_at       = ?
        WHERE id = ?
        """
        try:
            card.updated_at = datetime.now(timezone.utc)
            with self._connect() as conn:
                conn.execute(sql, (
                    card.interval,
                    card.ease_factor,
                    card.repetitions,
                    _fmt(card.next_review_date),
                    card.status,
                    _fmt(card.updated_at),
                    card.id,
                ))
            _log.debug(
                "Card updated | id=%d | interval=%d days | ease=%.2f | status=%s",
                card.id, card.interval, card.ease_factor, card.status,
            )
        except Exception:
            _log.exception("update_card failed | id=%s", card.id)
            raise

    def get_due_cards(self, limit: int = 20) -> list[Card]:
        """
        Return cards whose next_review_date is at or before now, ordered by
        urgency (most overdue first).  New cards are excluded — they are
        introduced via get_new_cards().
        """
        now = _fmt(datetime.now(timezone.utc))
        sql = """
        SELECT * FROM cards
        WHERE next_review_date <= ?
          AND status != 'new'
        ORDER BY next_review_date ASC
        LIMIT ?
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (now, limit)).fetchall()
            cards = [_row_to_card(r) for r in rows]
            _log.debug("Due cards fetched | count=%d", len(cards))
            return cards
        except Exception:
            _log.exception("get_due_cards failed")
            raise

    def get_new_cards(self, limit: int = 10) -> list[Card]:
        """Return unseen (status='new') cards for introducing new material."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM cards WHERE status = 'new' LIMIT ?", (limit,)
                ).fetchall()
            return [_row_to_card(r) for r in rows]
        except Exception:
            _log.exception("get_new_cards failed")
            raise

    # ── Review Log ────────────────────────────────────────────────────────────

    def add_review_log(self, entry: ReviewLog) -> ReviewLog:
        """Append one review record and return it with its generated id."""
        sql = """
        INSERT INTO review_log
            (card_id, grade, old_interval, new_interval,
             old_ease_factor, new_ease_factor, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, (
                    entry.card_id,
                    entry.grade,
                    entry.old_interval,
                    entry.new_interval,
                    entry.old_ease_factor,
                    entry.new_ease_factor,
                    _fmt(entry.reviewed_at),
                ))
                entry.id = cur.lastrowid
            _log.debug(
                "ReviewLog written | card=%d | grade=%d | %d→%d days | ease %.2f→%.2f",
                entry.card_id, entry.grade,
                entry.old_interval, entry.new_interval,
                entry.old_ease_factor, entry.new_ease_factor,
            )
            return entry
        except Exception:
            _log.exception("add_review_log failed | card_id=%d", entry.card_id)
            raise

    def get_review_history(self, card_id: int) -> list[ReviewLog]:
        """Return all review records for a card, ordered by insertion (id ASC)."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM review_log WHERE card_id = ? ORDER BY id ASC",
                    (card_id,),
                ).fetchall()
            return [_row_to_review_log(r) for r in rows]
        except Exception:
            _log.exception("get_review_history failed | card_id=%d", card_id)
            raise


# ── Row mappers (module-level so they can be tested independently) ─────────────

def _row_to_sentence(row: sqlite3.Row) -> Sentence:
    return Sentence(
        id=row["id"],
        sentence_en=row["sentence_en"],
        sentence_fa=row["sentence_fa"],
        context_notes=row["context_notes"],
        source=row["source"],
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


def _row_to_card(row: sqlite3.Row) -> Card:
    return Card(
        id=row["id"],
        sentence_id=row["sentence_id"],
        interval=row["interval"],
        ease_factor=row["ease_factor"],
        repetitions=row["repetitions"],
        next_review_date=_parse(row["next_review_date"]),
        status=row["status"],
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


def _row_to_review_log(row: sqlite3.Row) -> ReviewLog:
    return ReviewLog(
        id=row["id"],
        card_id=row["card_id"],
        grade=row["grade"],
        old_interval=row["old_interval"],
        new_interval=row["new_interval"],
        old_ease_factor=row["old_ease_factor"],
        new_ease_factor=row["new_ease_factor"],
        reviewed_at=_parse(row["reviewed_at"]),
    )
