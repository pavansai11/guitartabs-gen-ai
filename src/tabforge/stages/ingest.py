"""Stage 1 — Ingest.

Load audio, convert to mono, resample to 44100 Hz, peak-normalise to -1 dBFS.
Emit a stable ``song_id`` (sha256 of the file bytes, first 12 hex chars) used
for all cache and output paths.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config


@dataclass
class IngestResult:
    song_id: str
    audio: np.ndarray        # mono float32 in [-1, 1]
    sr: int
    source_path: Path
    wav_path: Path | None    # normalised wav written under out/<song_id>/


def compute_song_id(path: str | Path) -> str:
    """First 12 hex chars of the sha256 of the file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def peak_normalise(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    """Scale so the peak sits at ``peak_dbfs`` dBFS. No-op on silence."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=True)
    target = 10.0 ** (peak_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def ingest(path: str | Path, cfg: Config, out_root: str | Path = "out") -> IngestResult:
    """Load, mono, resample, normalise, and cache the processed wav."""
    try:
        import soundfile as sf
        import librosa
    except ImportError as exc:  # pragma: no cover - requires the audio extra
        raise ImportError(
            "Stage 1 (ingest) needs the audio extra: pip install 'tabforge[audio]'"
        ) from exc

    path = Path(path)
    song_id = compute_song_id(path)
    target_sr = int(cfg.get("audio", "target_sr", 44100))
    peak_dbfs = float(cfg.get("audio", "peak_dbfs", -1.0))

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # mono: average channels
    mono = audio.mean(axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    mono = peak_normalise(mono, peak_dbfs)

    song_dir = Path(out_root) / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    wav_path = song_dir / "ingest.wav"
    sf.write(str(wav_path), mono, sr)

    return IngestResult(
        song_id=song_id, audio=mono, sr=sr, source_path=path, wav_path=wav_path
    )
