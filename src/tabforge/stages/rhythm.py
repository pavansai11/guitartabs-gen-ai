"""Stage 7 — Rhythm.

librosa.beat.beat_track for tempo and beat grid. Quantise note starts to the
nearest 1/16 note, but ONLY when the shift is under max_shift_ms — larger
shifts mean the beat grid is wrong, and a wrong grid is worse than no grid.

--no-quantize bypasses entirely. Both raw and quantised times are stored; the
raw start (NoteEvent.raw_start_s) is never discarded. On low beat-tracking
confidence (rubato intros), quantisation falls back to off automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import Config
from ..models import NoteEvent


@dataclass
class BeatGrid:
    tempo_bpm: float
    beats_s: np.ndarray      # beat times in seconds
    confidence: float        # 0..1 regularity proxy


def estimate_beats(audio: np.ndarray, sr: int, cfg: Config) -> BeatGrid:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - requires the audio extra
        raise ImportError(
            "Stage 7 (rhythm) needs librosa: pip install 'tabforge[audio]'"
        ) from exc
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    return BeatGrid(tempo_bpm=tempo, beats_s=beats, confidence=beat_confidence(beats))


def beat_confidence(beats: np.ndarray) -> float:
    """Regularity proxy in [0,1]: steady inter-beat intervals -> high; rubato
    (highly variable intervals) -> low."""
    beats = np.asarray(beats, dtype=float)
    if beats.size < 3:
        return 0.0
    ibi = np.diff(beats)
    mean = float(np.mean(ibi))
    if mean <= 0:
        return 0.0
    cv = float(np.std(ibi) / mean)
    return float(max(0.0, min(1.0, 1.0 - cv)))


def build_sixteenth_grid(beats: np.ndarray, subdivisions: int = 4) -> np.ndarray:
    """Interpolate a 1/16-note grid between beats (4 sixteenths per quarter)."""
    beats = np.asarray(beats, dtype=float)
    if beats.size < 2:
        return beats
    grid = []
    for t0, t1 in zip(beats[:-1], beats[1:]):
        for k in range(subdivisions):
            grid.append(t0 + k * (t1 - t0) / subdivisions)
    grid.append(float(beats[-1]))
    return np.asarray(grid, dtype=float)


def quantize_notes(notes: list[NoteEvent], grid: np.ndarray,
                   cfg: Config) -> list[NoteEvent]:
    """Snap note starts to the nearest grid point when within max_shift_ms.

    raw_start_s is preserved on every note; start_s becomes the quantised time
    (or stays raw when the nearest grid point is too far)."""
    max_shift = float(cfg.get("rhythm", "max_shift_ms", 60.0)) / 1000.0
    grid = np.asarray(grid, dtype=float)
    out: list[NoteEvent] = []
    for n in notes:
        raw = n.raw_start_s if n.raw_start_s is not None else n.start_s
        if grid.size == 0:
            out.append(replace(n, raw_start_s=raw))
            continue
        j = int(np.argmin(np.abs(grid - raw)))
        target = float(grid[j])
        if abs(target - raw) <= max_shift:
            dur = n.end_s - n.start_s
            out.append(replace(n, start_s=target, end_s=target + dur, raw_start_s=raw))
        else:
            out.append(replace(n, raw_start_s=raw))
    return out


def apply_rhythm(notes: list[NoteEvent], audio: np.ndarray, sr: int, cfg: Config,
                 quantize: bool | None = None) -> tuple[list[NoteEvent], BeatGrid]:
    """Estimate the beat grid and (optionally) quantise. Returns (notes, grid)."""
    grid = estimate_beats(audio, sr, cfg)
    do_quant = cfg.get("rhythm", "quantize", True) if quantize is None else quantize
    min_conf = float(cfg.get("rhythm", "min_beat_confidence", 0.10))
    if do_quant and grid.confidence >= min_conf:
        sixteenths = build_sixteenth_grid(grid.beats_s)
        notes = quantize_notes(notes, sixteenths, cfg)
    else:
        # keep raw starts; still record raw_start_s for every note
        notes = [replace(n, raw_start_s=(n.raw_start_s if n.raw_start_s is not None
                                         else n.start_s)) for n in notes]
    return notes, grid
