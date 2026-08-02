"""
Raw PCM16 -> WAV wrapping for browser-recorded shadowing attempts.

This app always runs as a Flet web view (see main.py's ft.app() call), so
flet_audio_recorder's only way to get real audio bytes into this Python
process is to stream raw PCM16BITS data to a server-side upload endpoint —
the browser has no access to the server's filesystem to write a file
directly, and the client-side "path" it reports back is a blob: URL that
only exists inside the browser. This module turns those headerless PCM
bytes into a genuine playable/comparable WAV file once they land here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from logger_config import get_logger

_log = get_logger(__name__)

SAMPLE_RATE = 16000  # must match the AudioRecorderConfiguration used to record
CHANNELS = 1


def pcm16_bytes_to_wav(raw_bytes: bytes, output_path: Path) -> Path:
    """
    Wrap headerless 16-bit PCM audio bytes into a proper WAV file.

    Raises:
        ValueError: if `raw_bytes` is empty or not a whole number of int16
            samples (a truncated/corrupt upload).
    """
    if not raw_bytes:
        raise ValueError("No audio data to write — recording may have failed")
    if len(raw_bytes) % 2 != 0:
        raise ValueError("Raw audio byte count is not a whole number of 16-bit samples")

    samples = np.frombuffer(raw_bytes, dtype="<i2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), samples, SAMPLE_RATE, subtype="PCM_16")
    _log.info("Wrapped %d raw PCM bytes into WAV | path=%s", len(raw_bytes), output_path)
    return output_path
