"""
Waveform similarity scoring for the motor-memory shadowing exercise.

Deliberately NOT phoneme/ASR-based (too heavy for mobile — no PyTorch, no
speech-recognition model). Instead compares the RMS energy envelope of two
recordings — the rhythm/stress/timing pattern of speech — via normalized
cross-correlation after resampling both to the same length. This rewards
matching the cadence and emphasis of the reference, which is the actual
point of shadowing practice, using only numpy/scipy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from logger_config import get_logger

_log = get_logger(__name__)


def _load_mono(path: Path | str) -> tuple[int, np.ndarray]:
    """Load an audio file as mono float32 samples, whatever container format
    it's actually in (soundfile handles WAV/AIFF/FLAC/OGG transparently —
    important because some TTS backends label AIFF-C output with a .wav
    extension)."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return rate, mono


def _energy_envelope(samples: np.ndarray, rate: int, window_ms: int = 20) -> np.ndarray:
    """RMS energy per fixed-size window — a coarse rhythm/stress contour."""
    window = max(1, int(rate * window_ms / 1000))
    n_windows = len(samples) // window
    if n_windows == 0:
        return np.array([], dtype=np.float32)
    trimmed = samples[: n_windows * window]
    frames = trimmed.reshape(n_windows, window)
    return np.sqrt(np.mean(frames**2, axis=1))


def _normalize(vector: np.ndarray) -> np.ndarray:
    centered = vector - vector.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 1e-9 else centered


def score_shadowing(reference_path: Path | str, attempt_path: Path | str) -> float:
    """
    Compare a user's recorded shadowing attempt against a reference
    recording of the same sentence.

    Returns:
        A 0-100 similarity score. 100 means the two energy envelopes are
        perfectly correlated after time-normalisation; 0 means no
        correlation or below (uncorrelated/anti-correlated timing).

    Raises:
        ValueError: if either file has no usable audio at all.
    """
    ref_rate, ref_samples = _load_mono(reference_path)
    att_rate, att_samples = _load_mono(attempt_path)

    ref_env = _energy_envelope(ref_samples, ref_rate)
    att_env = _energy_envelope(att_samples, att_rate)

    if len(ref_env) == 0 or len(att_env) == 0:
        raise ValueError("Recording is too short to score")

    # Resample the attempt's envelope onto the reference's length so
    # differing recording durations don't zero out the correlation.
    att_env_resampled = resample(att_env, len(ref_env))

    ref_norm = _normalize(ref_env)
    att_norm = _normalize(np.asarray(att_env_resampled))

    if np.linalg.norm(ref_norm) < 1e-9 or np.linalg.norm(att_norm) < 1e-9:
        # One of the two envelopes is silent/flat — no rhythm to compare.
        return 0.0

    cosine_similarity = float(np.dot(ref_norm, att_norm))
    score = max(0.0, min(100.0, (cosine_similarity + 1) / 2 * 100))
    _log.debug(
        "Shadowing score computed | reference=%s | attempt=%s | score=%.1f",
        reference_path, attempt_path, score,
    )
    return round(score, 1)
