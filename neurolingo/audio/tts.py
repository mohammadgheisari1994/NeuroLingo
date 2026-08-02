"""
Text-to-speech reference audio for the motor-memory shadowing exercise.

Uses pyttsx3 — a thin wrapper around the OS's native speech synthesizer
(NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux) — rather
than a bundled ML model, keeping this mobile-safe: no PyTorch, no downloaded
voice weights, just whatever TTS engine the OS already ships.

Synthesized clips are cached to disk keyed by a hash of the sentence text,
so shadowing the same sentence twice doesn't re-synthesize.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import soundfile as sf

from logger_config import get_logger

_log = get_logger(__name__)

try:
    import pyttsx3

    _AVAILABLE = True
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]
    _AVAILABLE = False


def is_available() -> bool:
    """True if pyttsx3 (and a usable OS voice) is present."""
    if not _AVAILABLE:
        return False
    try:
        engine = pyttsx3.init()
        has_voice = bool(engine.getProperty("voices"))
        engine.stop()
        return has_voice
    except Exception:
        _log.exception("TTS availability check failed")
        return False


def _cache_path(cache_dir: Path, text: str) -> Path:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.wav"


def get_reference_audio(text: str, cache_dir: Path) -> Path:
    """
    Return a path to a WAV file of `text` spoken aloud, synthesizing and
    caching it on first use.

    Some TTS backends (macOS's NSSpeechSynthesizer, via pyttsx3) label
    their native AIFF-C output with a .wav extension regardless of what
    was requested — re-saved here through soundfile so the cached file is
    genuinely WAV-encoded and playable/comparable everywhere downstream.

    Raises:
        RuntimeError: if pyttsx3 isn't installed, has no usable voice, or
            fails to produce a file.
    """
    if not _AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(cache_dir, text)
    if cached.exists():
        return cached

    raw_path = cached.with_suffix(".raw.wav")
    engine = pyttsx3.init()
    try:
        engine.save_to_file(text, str(raw_path))
        engine.runAndWait()
    finally:
        engine.stop()

    if not raw_path.exists():
        raise RuntimeError("TTS engine did not produce an audio file")

    try:
        data, rate = sf.read(str(raw_path))
        sf.write(str(cached), data, rate)
    finally:
        raw_path.unlink(missing_ok=True)

    _log.info("Synthesized reference audio | text=%.40s | path=%s", text, cached)
    return cached
