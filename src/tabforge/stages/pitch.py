"""Stage 3 — Pitch tracking.

torchcrepe (full model) at a 5 ms hop, fmin 65 Hz, fmax 1400 Hz. The small hop
is essential: a 60 ms slide is only 6 frames at a 10 ms hop, too few to fit a
slope reliably.

Median-filter (window 3) to remove single-frame octave errors, then mask frames
with confidence < 0.5 as unvoiced. Do NOT smooth further — aggressive smoothing
destroys vibrato.

Falls back to librosa.pyin when torchcrepe is unavailable.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..dsp import median_filter_1d
from ..models import F0Track


def _track_torchcrepe(audio: np.ndarray, sr: int, cfg: Config) -> F0Track | None:
    try:
        import torch
        import torchcrepe
    except ImportError:
        return None

    hop_s = float(cfg.get("pitch", "hop_ms", 5.0)) / 1000.0
    hop_length = max(1, int(round(hop_s * sr)))
    fmin = float(cfg.get("pitch", "fmin", 65.0))
    fmax = float(cfg.get("pitch", "fmax", 1400.0))
    model = cfg.get("pitch", "model", "full")
    conf_thresh = float(cfg.get("pitch", "confidence_threshold", 0.5))
    med_win = int(cfg.get("pitch", "median_window", 3))

    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    pitch, periodicity = torchcrepe.predict(
        tensor, sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
        model=model, return_periodicity=True, batch_size=512,
    )
    # median filter kills single-frame octave errors
    pitch = torchcrepe.filter.median(pitch, med_win)
    periodicity = torchcrepe.filter.median(periodicity, med_win)

    hz = pitch.squeeze(0).cpu().numpy().astype(float)
    conf = periodicity.squeeze(0).cpu().numpy().astype(float)
    hz[conf < conf_thresh] = 0.0

    n = len(hz)
    times = np.arange(n) * hop_s
    return F0Track(times=times, hz=hz, confidence=conf, hop_s=hop_s)


def _track_pyin(audio: np.ndarray, sr: int, cfg: Config) -> F0Track:
    import librosa

    hop_s = float(cfg.get("pitch", "hop_ms", 5.0)) / 1000.0
    hop_length = max(1, int(round(hop_s * sr)))
    fmin = float(cfg.get("pitch", "fmin", 65.0))
    fmax = float(cfg.get("pitch", "fmax", 1400.0))
    conf_thresh = float(cfg.get("pitch", "confidence_threshold", 0.5))
    med_win = int(cfg.get("pitch", "median_window", 3))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
        frame_length=max(2048, hop_length * 4),
    )
    hz = np.nan_to_num(f0, nan=0.0).astype(float)
    hz = median_filter_1d(np.where(hz > 0, hz, np.nan), med_win)
    hz = np.nan_to_num(hz, nan=0.0)
    conf = np.nan_to_num(voiced_prob, nan=0.0).astype(float)
    hz[conf < conf_thresh] = 0.0

    n = len(hz)
    times = np.arange(n) * hop_s
    return F0Track(times=times, hz=hz, confidence=conf, hop_s=hop_s)


def track_pitch(audio: np.ndarray, sr: int, cfg: Config) -> F0Track:
    """Extract the f0 contour, preferring torchcrepe and falling back to pyin."""
    result = _track_torchcrepe(audio, sr, cfg)
    if result is not None:
        return result
    try:
        return _track_pyin(audio, sr, cfg)
    except ImportError as exc:  # pragma: no cover - requires the audio extra
        raise ImportError(
            "Stage 3 (pitch) needs torchcrepe or librosa: "
            "pip install 'tabforge[audio]'"
        ) from exc
