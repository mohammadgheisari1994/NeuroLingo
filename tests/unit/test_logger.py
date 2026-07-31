"""
Unit tests for logger_config.py.

These tests cover:
- Logger identity and type
- Handler count idempotency
- Log-file creation
- Child logger naming
- Level propagation via caplog
- Custom log level
"""
import logging

import pytest

from logger_config import LOG_FILE, get_logger, setup_logging

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_logging():
    """Guarantee a clean root logger before and after every test."""
    yield
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_returns_neurolingo_logger():
    logger = setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "neurolingo"


def test_root_has_exactly_two_handlers():
    setup_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 2


def test_no_duplicate_handlers_on_repeated_calls():
    for _ in range(3):
        setup_logging()
    root = logging.getLogger()
    assert len(root.handlers) == 2


def test_log_file_is_created():
    setup_logging()
    assert LOG_FILE.exists(), f"Expected log file at {LOG_FILE}"


def test_get_logger_produces_child_namespace():
    setup_logging()
    child = get_logger("some.module")
    assert child.name == "neurolingo.some.module"


def test_all_levels_written_to_file():
    """Verify all 4 log levels reach the rotating file handler."""
    import uuid

    run_id = uuid.uuid4().hex[:8]
    logger = setup_logging(level="DEBUG")
    logger.debug("dbg-%s", run_id)
    logger.info("info-%s", run_id)
    logger.warning("warn-%s", run_id)
    logger.error("err-%s", run_id)

    # Flush before reading
    for h in logging.getLogger().handlers:
        h.flush()

    content = LOG_FILE.read_text(encoding="utf-8")
    assert f"dbg-{run_id}" in content
    assert f"info-{run_id}" in content
    assert f"warn-{run_id}" in content
    assert f"err-{run_id}" in content


def test_custom_level_applied():
    setup_logging(level="WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_file_handler_level_is_debug():
    setup_logging()
    root = logging.getLogger()
    file_handler = next(
        h for h in root.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    )
    assert file_handler.level == logging.DEBUG


def test_console_handler_level_is_info():
    setup_logging()
    root = logging.getLogger()
    console_handler = next(
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
    )
    assert console_handler.level == logging.INFO
