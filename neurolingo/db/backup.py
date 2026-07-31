"""
JSON export/import for full user data backup (sentences + SRS cards +
review history).

Deliberately a portable format, not a raw table dump: exported entries never
include internal database ids, and import always inserts fresh rows via the
same public repository methods the rest of the app uses — so a backup can be
restored into an empty database, merged into an existing one, or moved
between devices without id collisions. Restoring the same backup twice
duplicates sentences; there is no de-duplication, matching "restore a
backup" rather than "sync" semantics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from logger_config import get_logger
from neurolingo.db.models import Card, ReviewLog, Sentence
from neurolingo.db.repository import DatabaseRepository

_log = get_logger(__name__)

BACKUP_FORMAT_VERSION = 1


class BackupFormatError(Exception):
    """Raised when import_backup() is given data it can't understand."""


def export_backup(repo: DatabaseRepository) -> dict[str, Any]:
    """Serialise every sentence, its card, and its review history into a
    portable dict, ready for json.dumps()."""
    sentences_out: list[dict[str, Any]] = []

    for sentence in repo.get_all_sentences():
        entry: dict[str, Any] = {
            "sentence_en": sentence.sentence_en,
            "sentence_fa": sentence.sentence_fa,
            "context_notes": sentence.context_notes,
            "source": sentence.source,
            "card": None,
            "review_log": [],
        }

        card = repo.get_card_by_sentence_id(sentence.id)
        if card is not None:
            entry["card"] = {
                "interval": card.interval,
                "ease_factor": card.ease_factor,
                "repetitions": card.repetitions,
                "next_review_date": card.next_review_date.isoformat(),
                "status": card.status,
            }
            entry["review_log"] = [
                {
                    "grade": r.grade,
                    "old_interval": r.old_interval,
                    "new_interval": r.new_interval,
                    "old_ease_factor": r.old_ease_factor,
                    "new_ease_factor": r.new_ease_factor,
                    "reviewed_at": r.reviewed_at.isoformat(),
                }
                for r in repo.get_review_history(card.id)
            ]

        sentences_out.append(entry)

    backup = {
        "format_version": BACKUP_FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sentences": sentences_out,
    }
    _log.info("Exported backup | sentences=%d", len(sentences_out))
    return backup


def import_backup(repo: DatabaseRepository, data: dict[str, Any]) -> int:
    """
    Restore sentences (+ cards + review history) from a backup dict produced
    by export_backup(). Always inserts fresh rows.

    Returns:
        The number of sentences imported.

    Raises:
        BackupFormatError: `data` isn't a recognised backup (wrong shape,
            unsupported format_version).
    """
    if not isinstance(data, dict) or "sentences" not in data:
        raise BackupFormatError("Not a NeuroLingo backup file (missing 'sentences' key)")

    version = data.get("format_version")
    if version != BACKUP_FORMAT_VERSION:
        raise BackupFormatError(
            f"Unsupported backup format_version {version!r} "
            f"(this NeuroLingo only understands version {BACKUP_FORMAT_VERSION})"
        )

    imported = 0
    for entry in data["sentences"]:
        sentence = repo.add_sentence(Sentence(
            sentence_en=entry["sentence_en"],
            sentence_fa=entry["sentence_fa"],
            context_notes=entry.get("context_notes", ""),
            source=entry.get("source", ""),
        ))
        imported += 1

        card_data = entry.get("card")
        if card_data is None:
            continue

        card = repo.add_card(Card(
            sentence_id=sentence.id,
            interval=card_data["interval"],
            ease_factor=card_data["ease_factor"],
            repetitions=card_data["repetitions"],
            next_review_date=datetime.fromisoformat(card_data["next_review_date"]),
            status=card_data["status"],
        ))

        for log_entry in entry.get("review_log", []):
            repo.add_review_log(ReviewLog(
                card_id=card.id,
                grade=log_entry["grade"],
                old_interval=log_entry["old_interval"],
                new_interval=log_entry["new_interval"],
                old_ease_factor=log_entry["old_ease_factor"],
                new_ease_factor=log_entry["new_ease_factor"],
                reviewed_at=datetime.fromisoformat(log_entry["reviewed_at"]),
            ))

    _log.info("Imported backup | sentences=%d", imported)
    return imported
