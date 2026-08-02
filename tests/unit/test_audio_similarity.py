"""
Unit tests for neurolingo.audio.similarity.

All fixtures are synthetic sine-wave "pulses" written to real WAV files via
soundfile (no microphone/TTS engine needed — this module only ever touches
files, never hardware), so these run identically in CI.

Coverage targets:
- Identical audio scores near-perfectly
- Same rhythm at a different pitch/duration still scores high (envelope-
  based, not phoneme-based, by design)
- A completely different rhythm scores low
- Silence / too-short audio raises rather than crashing
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from neurolingo.audio.similarity import score_shadowing

RATE = 16000


def _pulses(pattern: list[float], pulse_ms: int = 150, gap_ms: int = 150, freq: float = 220.0):
    """Build a waveform of tone pulses at given amplitudes, separated by
    silence — a crude but effective stand-in for "stressed syllable, gap,
    stressed syllable" speech rhythm."""
    pulse_n = int(RATE * pulse_ms / 1000)
    gap_n = int(RATE * gap_ms / 1000)
    t = np.arange(pulse_n) / RATE
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    silence = np.zeros(gap_n, dtype=np.float32)

    chunks = []
    for amplitude in pattern:
        chunks.append(tone * amplitude)
        chunks.append(silence)
    return np.concatenate(chunks)


def _write_wav(path, samples):
    sf.write(str(path), samples, RATE)
    return path


@pytest.fixture
def reference_wav(tmp_path):
    # Loud-soft-loud rhythm — like a stressed-unstressed-stressed sentence.
    return _write_wav(tmp_path / "reference.wav", _pulses([1.0, 0.2, 1.0]))


def test_identical_audio_scores_near_perfect(reference_wav, tmp_path):
    attempt = _write_wav(tmp_path / "attempt.wav", _pulses([1.0, 0.2, 1.0]))
    score = score_shadowing(reference_wav, attempt)
    assert score > 95.0


def test_same_rhythm_different_pitch_still_scores_high(reference_wav, tmp_path):
    """Envelope-based comparison should be pitch-agnostic — a different
    voice/frequency saying the same rhythm should still score well."""
    attempt = _write_wav(tmp_path / "attempt.wav", _pulses([1.0, 0.2, 1.0], freq=440.0))
    score = score_shadowing(reference_wav, attempt)
    assert score > 85.0


def test_same_rhythm_different_duration_still_scores_reasonably(reference_wav, tmp_path):
    """Resampling should absorb a faster/slower attempt at the same
    relative rhythm shape."""
    attempt = _write_wav(
        tmp_path / "attempt.wav",
        _pulses([1.0, 0.2, 1.0], pulse_ms=100, gap_ms=100),
    )
    score = score_shadowing(reference_wav, attempt)
    assert score > 70.0


def test_different_rhythm_scores_clearly_worse_than_a_match(reference_wav, tmp_path):
    """A single continuous tone (no rhythm at all) is a qualitatively
    different envelope shape from the reference's three loud-soft-loud
    pulses — should score well below the 95+ a real match gets."""
    steady_tone = np.sin(2 * np.pi * 220.0 * np.arange(RATE) / RATE).astype(np.float32)
    attempt = _write_wav(tmp_path / "attempt.wav", steady_tone)
    score = score_shadowing(reference_wav, attempt)
    assert score < 70.0


def test_silence_scores_zero_not_a_crash(reference_wav, tmp_path):
    attempt = _write_wav(tmp_path / "attempt.wav", np.zeros(RATE, dtype=np.float32))
    score = score_shadowing(reference_wav, attempt)
    assert score == 0.0


def test_too_short_recording_raises(reference_wav, tmp_path):
    attempt = _write_wav(tmp_path / "attempt.wav", np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="too short"):
        score_shadowing(reference_wav, attempt)


def test_score_is_stable_and_bounded(reference_wav, tmp_path):
    attempt = _write_wav(tmp_path / "attempt.wav", _pulses([0.9, 0.15, 1.0]))
    score = score_shadowing(reference_wav, attempt)
    assert 0.0 <= score <= 100.0
