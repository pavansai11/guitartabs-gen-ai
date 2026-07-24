"""Stage 4 — Onset detection.

librosa.onset.onset_detect with backtrack=True on the separated stem, using
spectral flux. Onsets mark PLUCKS — the single most important discriminator in
the system:

    A pitch change WITH a coincident onset is a new picked note.
    A pitch change WITHOUT one is an articulation.

Prefer slightly over-sensitive: a false onset produces a wrong "pluck" label,
which is more visible and easier to correct than a missed note.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import Onset


def detect_onsets(audio: np.ndarray, sr: int, cfg: Config) -> list[Onset]:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - requires the audio extra
        raise ImportError(
            "Stage 4 (onsets) needs librosa: pip install 'tabforge[audio]'"
        ) from exc

    hop_length = 512
    backtrack = bool(cfg.get("onsets", "backtrack", True))
    delta = float(cfg.get("onsets", "delta", 0.06))

    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop_length,
        backtrack=backtrack, delta=delta, units="frames",
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    onsets = []
    for f, t in zip(frames, times):
        strength = float(env[f]) if 0 <= f < len(env) else 0.0
        onsets.append(Onset(time_s=float(t), strength=strength))
    return onsets


def onset_near(onsets: list[Onset], time_s: float, window_s: float) -> Onset | None:
    """Return the closest onset within +/- window_s of time_s, or None."""
    best, best_dt = None, window_s
    for o in onsets:
        dt = abs(o.time_s - time_s)
        if dt <= best_dt:
            best, best_dt = o, dt
    return best
