"""
Unit tests for neurolingo.audio.tts.

pyttsx3 needs a real OS speech engine, unavailable on CI runners — these
tests mock the pyttsx3 module so the caching/format-normalisation logic
(the actual code this project owns) is verified deterministically
everywhere, while a real end-to-end synthesis was manually verified once
against the live macOS TTS engine during development (see PR description).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

import neurolingo.audio.tts as tts


@pytest.fixture
def fake_pyttsx3(monkeypatch, tmp_path):
    """A pyttsx3 stand-in whose save_to_file() writes a real (tiny) WAV file,
    so the downstream soundfile re-save step has something genuine to read —
    this is what actually exercises the AIFF-mislabelled-as-wav fix."""
    written_texts = []

    def _save_to_file(text, path):
        written_texts.append(text)
        tone = np.sin(2 * np.pi * 220 * np.arange(8000) / 8000).astype(np.float32)
        sf.write(path, tone, 8000)

    engine = MagicMock()
    engine.getProperty.return_value = ["some-voice"]
    engine.save_to_file.side_effect = _save_to_file

    module = MagicMock()
    module.init.return_value = engine

    monkeypatch.setattr(tts, "pyttsx3", module)
    monkeypatch.setattr(tts, "_AVAILABLE", True)
    return module, engine, written_texts


def test_is_available_true_when_engine_has_voices(fake_pyttsx3):
    assert tts.is_available() is True


def test_is_available_false_when_no_voices(monkeypatch, fake_pyttsx3):
    module, engine, _ = fake_pyttsx3
    engine.getProperty.return_value = []
    assert tts.is_available() is False


def test_is_available_false_when_not_installed(monkeypatch):
    monkeypatch.setattr(tts, "_AVAILABLE", False)
    assert tts.is_available() is False


def test_get_reference_audio_raises_when_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="not installed"):
        tts.get_reference_audio("Hello.", tmp_path)


def test_get_reference_audio_synthesizes_and_caches(fake_pyttsx3, tmp_path):
    _module, _engine, written_texts = fake_pyttsx3

    path = tts.get_reference_audio("She has been waiting for the bus.", tmp_path)

    assert path.exists()
    assert written_texts == ["She has been waiting for the bus."]
    # Output must be genuinely readable as WAV (the whole point of the
    # soundfile re-save step), not just whatever format the engine produced.
    data, rate = sf.read(str(path))
    assert rate == 8000
    assert len(data) > 0


def test_get_reference_audio_second_call_uses_cache(fake_pyttsx3, tmp_path):
    _module, _engine, written_texts = fake_pyttsx3

    first = tts.get_reference_audio("Same sentence.", tmp_path)
    second = tts.get_reference_audio("Same sentence.", tmp_path)

    assert first == second
    assert written_texts == ["Same sentence."]  # engine only invoked once


def test_different_sentences_get_different_cache_files(fake_pyttsx3, tmp_path):
    first = tts.get_reference_audio("Sentence one.", tmp_path)
    second = tts.get_reference_audio("Sentence two.", tmp_path)
    assert first != second


def test_raises_when_engine_produces_no_file(monkeypatch, tmp_path):
    engine = MagicMock()
    engine.getProperty.return_value = ["voice"]
    engine.save_to_file.side_effect = lambda text, path: None  # writes nothing

    module = MagicMock()
    module.init.return_value = engine
    monkeypatch.setattr(tts, "pyttsx3", module)
    monkeypatch.setattr(tts, "_AVAILABLE", True)

    with pytest.raises(RuntimeError, match="did not produce"):
        tts.get_reference_audio("Nothing will be written.", tmp_path)
