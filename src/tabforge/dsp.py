"""Shared numeric helpers used across pitch/segment/articulation stages.

Kept dependency-light (numpy only) so the differentiating logic runs and is
testable without the heavy audio stack.
"""

from __future__ import annotations

import numpy as np


def hz_to_cents(hz: np.ndarray | float) -> np.ndarray:
    """Continuous pitch in cents: 1200*log2(hz/440)+6900.

    cents/100 == MIDI note number. Returns NaN where hz <= 0 (unvoiced).
    """
    hz = np.asarray(hz, dtype=float)
    out = np.full(hz.shape, np.nan, dtype=float)
    voiced = hz > 0
    out[voiced] = 1200.0 * np.log2(hz[voiced] / 440.0) + 6900.0
    return out


def cents_to_hz(cents: np.ndarray | float) -> np.ndarray:
    cents = np.asarray(cents, dtype=float)
    return 440.0 * np.power(2.0, (cents - 6900.0) / 1200.0)


def cents_to_midi(cents: float) -> int:
    """Nearest MIDI note number for a cents value (cents/100 rounded)."""
    return int(round(cents / 100.0))


def midi_to_cents(midi: int) -> float:
    return float(midi) * 100.0


def linfit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line fit. Returns (slope, intercept, r2).

    r2 is the coefficient of determination in [0, 1]; if y has no variance it
    is defined as 1.0 (a flat line fits a flat signal perfectly).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def longest_stable_run(cents: np.ndarray, tol_cents: float) -> tuple[int, int]:
    """Longest contiguous run of frames staying within +/- tol of its own mean.

    Returns (start_index, end_index_exclusive). Ignores NaN frames (they break
    a run). Used to find the "stable portion" of a note so bends/slides do not
    drag the nominal pitch off.
    """
    n = len(cents)
    if n == 0:
        return 0, 0
    best_start, best_end = 0, 0
    i = 0
    while i < n:
        if np.isnan(cents[i]):
            i += 1
            continue
        j = i + 1
        # grow window while every frame stays within tol of the running mean
        while j < n and not np.isnan(cents[j]):
            window = cents[i:j + 1]
            m = float(np.mean(window))
            if np.max(np.abs(window - m)) <= tol_cents:
                j += 1
            else:
                break
        if (j - i) > (best_end - best_start):
            best_start, best_end = i, j
        i = j if j > i else i + 1
    return best_start, best_end


def median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Odd-window median filter that ignores NaN, used to kill single-frame
    octave errors without pulling in the heavy scipy dependency at call sites
    that only have numpy."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = x.copy()
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out
