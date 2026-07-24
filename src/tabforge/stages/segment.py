"""Stage 5 — Note segmentation.

Walk the cents contour and cut a new note event when either:
  - an onset occurs, or
  - the contour holds a NEW semitone band (within +/- new_band_cents of a
    different semitone) for at least new_band_hold_ms.

Discard events shorter than min_note_ms. For each event compute ``midi`` from
the median cents of its STABLE portion (the longest run staying within
+/- stable_tol_cents), not the mean of the whole event — otherwise a bend or
slide drags the nominal pitch off.

Pure numpy: runs and is tested without any audio stack.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..dsp import cents_to_midi, longest_stable_run
from ..models import F0Track, NoteEvent, Onset


def _voiced_runs(cents: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where cents is not NaN."""
    runs = []
    n = len(cents)
    i = 0
    while i < n:
        if np.isnan(cents[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(cents[j]):
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _band_change_cuts(cents: np.ndarray, run: tuple[int, int],
                      hold_frames: int, band_tol: float) -> list[int]:
    """Indices within [run) where a new semitone band starts and is held."""
    s, e = run
    cuts: list[int] = []
    band = int(round(cents[s] / 100.0))
    i = s
    while i < e:
        b = int(round(cents[i] / 100.0))
        if b != band:
            hi = min(e, i + hold_frames)
            window = cents[i:hi]
            held = (len(window) >= hold_frames
                    and np.all(np.abs(window - b * 100.0) <= band_tol))
            if held:
                cuts.append(i)
                band = b
                i = hi
                continue
        i += 1
    return cuts


def segment(f0: F0Track, onsets: list[Onset], cfg: Config) -> list[NoteEvent]:
    cents = f0.cents
    times = np.asarray(f0.times, dtype=float)
    conf = np.asarray(f0.confidence, dtype=float)
    hop_s = float(f0.hop_s)

    min_note_ms = float(cfg.get("segment", "min_note_ms", 45.0))
    band_tol = float(cfg.get("segment", "new_band_cents", 40.0))
    hold_ms = float(cfg.get("segment", "new_band_hold_ms", 40.0))
    stable_tol = float(cfg.get("segment", "stable_tol_cents", 35.0))
    hold_frames = max(1, math.ceil((hold_ms / 1000.0) / hop_s))

    onset_frames = sorted({int(round(o.time_s / hop_s)) for o in onsets})

    events: list[NoteEvent] = []
    for run in _voiced_runs(cents):
        s, e = run
        cuts = {s}
        cuts.update(f for f in onset_frames if s < f < e)
        cuts.update(_band_change_cuts(cents, run, hold_frames, band_tol))
        ordered = sorted(cuts) + [e]

        for a, b in zip(ordered[:-1], ordered[1:]):
            if b <= a:
                continue
            seg_cents = cents[a:b]
            dur_s = (b - a) * hop_s
            if dur_s * 1000.0 < min_note_ms:
                continue
            ss, se = longest_stable_run(seg_cents, stable_tol)
            stable = seg_cents[ss:se]
            stable = stable[~np.isnan(stable)]
            if stable.size == 0:
                stable = seg_cents[~np.isnan(seg_cents)]
            if stable.size == 0:
                continue
            median_cents = float(np.median(stable))
            midi = cents_to_midi(median_cents)
            cents_offset = median_cents - midi * 100.0
            seg_conf = conf[a:b]
            confidence = float(np.mean(seg_conf)) if seg_conf.size else 0.0
            start_s = float(times[a])
            end_s = float(times[b - 1] + hop_s)
            events.append(NoteEvent(
                start_s=start_s, end_s=end_s, midi=midi,
                cents_offset=cents_offset, confidence=confidence,
                raw_start_s=start_s,
            ))
    events.sort(key=lambda n: n.start_s)
    return events
