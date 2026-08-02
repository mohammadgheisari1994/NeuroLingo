"""Unit tests for neurolingo.audio.recorder (raw PCM16 -> WAV wrapping)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from neurolingo.audio.recorder import CHANNELS, SAMPLE_RATE, pcm16_bytes_to_wav


def _pcm16_bytes(seconds: float) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    tone = (np.sin(2 * np.pi * 220 * np.arange(n) / SAMPLE_RATE) * 30000).astype("<i2")
    return tone.tobytes()


def test_wraps_pcm_bytes_into_readable_wav(tmp_path):
    raw = _pcm16_bytes(1.0)
    out = pcm16_bytes_to_wav(raw, tmp_path / "attempt.wav")

    assert out.exists()
    data, rate = sf.read(str(out))
    assert rate == SAMPLE_RATE
    assert len(data) == SAMPLE_RATE  # 1 second at SAMPLE_RATE


def test_creates_parent_directories(tmp_path):
    out = pcm16_bytes_to_wav(_pcm16_bytes(0.1), tmp_path / "nested" / "dir" / "attempt.wav")
    assert out.exists()


def test_empty_bytes_raises(tmp_path):
    with pytest.raises(ValueError, match="No audio data"):
        pcm16_bytes_to_wav(b"", tmp_path / "attempt.wav")


def test_odd_byte_count_raises(tmp_path):
    with pytest.raises(ValueError, match="whole number"):
        pcm16_bytes_to_wav(b"\x01\x02\x03", tmp_path / "attempt.wav")


def test_output_is_mono(tmp_path):
    out = pcm16_bytes_to_wav(_pcm16_bytes(0.2), tmp_path / "attempt.wav")
    info = sf.info(str(out))
    assert info.channels == CHANNELS
