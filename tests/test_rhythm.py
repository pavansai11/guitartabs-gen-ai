"""Rhythm quantisation (Stage 7): pure grid + snap behaviour."""

from __future__ import annotations

import numpy as np

from tabforge.config import load_config
from tabforge.stages.rhythm import (beat_confidence, build_sixteenth_grid,
                                    quantize_notes)
from synth import note


def test_beat_confidence_high_for_steady_grid():
    steady = np.arange(0, 4, 0.5)          # perfectly regular
    assert beat_confidence(steady) > 0.9


def test_beat_confidence_low_for_rubato():
    rubato = np.array([0.0, 0.4, 1.3, 1.5, 2.9])   # wildly uneven
    assert beat_confidence(rubato) < 0.6


def test_sixteenth_grid_subdivides_beats():
    beats = np.array([0.0, 0.5, 1.0])      # 120 bpm quarter notes
    grid = build_sixteenth_grid(beats)
    # 4 sixteenths per quarter -> 0.125 s spacing
    assert np.allclose(np.diff(grid)[:4], 0.125)


def test_quantize_snaps_within_tolerance():
    cfg = load_config()
    grid = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
    n = note(0.13, 0.30, 64)               # 5 ms from the 0.125 grid point
    out = quantize_notes([n], grid, cfg)
    assert abs(out[0].start_s - 0.125) < 1e-9
    assert out[0].raw_start_s == 0.13      # raw is never discarded


def test_quantize_keeps_raw_when_shift_too_large():
    cfg = load_config()
    grid = np.array([0.0, 0.5, 1.0])       # nearest point 0.2 s away
    n = note(0.30, 0.50, 64)
    out = quantize_notes([n], grid, cfg)   # 200 ms > 60 ms tolerance
    assert out[0].start_s == 0.30          # unchanged
    assert out[0].raw_start_s == 0.30
