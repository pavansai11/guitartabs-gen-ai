"""Synthesised test signals.

M3 mandates validating articulation logic on numpy-generated signals with known
slides/bends/vibrato before touching real audio. These helpers build a cents
contour from piecewise segments and wrap it as an F0Track.
"""

from __future__ import annotations

import numpy as np

from tabforge.dsp import cents_to_hz
from tabforge.models import F0Track, NoteEvent

HOP_S = 0.005


def _n(dur_s: float, hop_s: float) -> int:
    return max(1, int(round(dur_s / hop_s)))


def ramp(dur_s: float, c0: float, c1: float, hop_s: float = HOP_S) -> np.ndarray:
    return np.linspace(c0, c1, _n(dur_s, hop_s))


def hold(dur_s: float, c: float, hop_s: float = HOP_S) -> np.ndarray:
    return np.full(_n(dur_s, hop_s), float(c))


def rest(dur_s: float, hop_s: float = HOP_S) -> np.ndarray:
    return np.full(_n(dur_s, hop_s), np.nan)


def vibrato(dur_s: float, center: float, freq_hz: float, depth_cents: float,
            hop_s: float = HOP_S) -> np.ndarray:
    n = _n(dur_s, hop_s)
    t = np.arange(n) * hop_s
    return center + (depth_cents / 2.0) * np.sin(2 * np.pi * freq_hz * t)


def track(*segments: np.ndarray, hop_s: float = HOP_S) -> F0Track:
    cents = np.concatenate(segments)
    filled = np.nan_to_num(cents, nan=440.0)
    hz = cents_to_hz(filled)
    hz = np.where(np.isnan(cents), 0.0, hz)
    conf = np.where(hz > 0, 1.0, 0.0)
    times = np.arange(len(cents)) * hop_s
    return F0Track(times=times, hz=hz, confidence=conf, hop_s=hop_s)


def note(start_s: float, end_s: float, midi: int, cents_offset: float = 0.0,
         **kw) -> NoteEvent:
    return NoteEvent(start_s=start_s, end_s=end_s, midi=midi,
                     cents_offset=cents_offset, confidence=1.0,
                     raw_start_s=start_s, **kw)
