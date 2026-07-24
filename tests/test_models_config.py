"""Data-model serialisation and config/preset loading."""

from __future__ import annotations

import numpy as np

from tabforge.config import get_tuning, load_config, load_tunings
from tabforge.models import (Articulation, BendCurve, F0Track, Score, TabNote,
                             dict_to_note, dict_to_score, note_to_dict,
                             score_to_dict)


def test_f0track_cents_and_roundtrip():
    hz = np.array([0.0, 440.0, 880.0])
    f0 = F0Track(times=np.array([0.0, 0.005, 0.01]), hz=hz,
                 confidence=np.array([0.0, 1.0, 1.0]), hop_s=0.005)
    cents = f0.cents
    assert np.isnan(cents[0])                 # unvoiced -> NaN
    assert abs(cents[1] - 6900.0) < 1e-6      # A4 = 6900 cents
    assert abs(cents[2] - 8100.0) < 1e-6      # A5 = one octave up
    back = F0Track.from_json_obj(f0.to_json_obj())
    assert np.allclose(back.hz, hz)


def test_tabnote_roundtrip_with_bend_and_vibrato():
    tn = TabNote(start_s=0.0, end_s=0.5, midi=64, cents_offset=3.0, confidence=0.9,
                 entry=Articulation.SLIDE_IN, exit=Articulation.RELEASE,
                 bend=BendCurve(peak_cents=200.0, onset_s=0.1, hold_s=0.1, released=True),
                 vibrato_hz=5.5, vibrato_depth_cents=40.0, raw_start_s=0.0,
                 string=1, fret=5)
    obj = note_to_dict(tn)
    back = dict_to_note(obj)
    assert isinstance(back, TabNote)
    assert back.string == 1 and back.fret == 5
    assert back.entry == Articulation.SLIDE_IN
    assert back.exit == Articulation.RELEASE
    assert back.bend is not None and back.bend.released is True
    assert back.vibrato_hz == 5.5


def test_score_roundtrip():
    tn = TabNote(start_s=0.0, end_s=0.5, midi=64, cents_offset=0.0, confidence=1.0,
                 string=0, fret=0)
    sc = Score(notes=[tn], tempo_bpm=120.0, time_signature=(4, 4),
               tuning=get_tuning("standard"), capo=2)
    back = dict_to_score(score_to_dict(sc))
    assert back.tempo_bpm == 120.0
    assert back.time_signature == (4, 4)
    assert back.capo == 2
    assert back.notes[0].fret == 0


def test_config_default_and_indian_preset():
    default = load_config()
    assert default.get("articulation", "slide_max_ms") == 250.0
    assert default.get("articulation", "slide_r2_min") == 0.85

    indian = load_config(preset="indian")
    assert indian.get("articulation", "slide_max_ms") == 400.0
    assert indian.get("articulation", "slide_r2_min") == 0.75
    # untouched keys survive the overlay
    assert indian.get("articulation", "bend_min_cents") == 80.0


def test_tunings():
    tunings = load_tunings()
    assert tunings["standard"] == [64, 59, 55, 50, 45, 40]
    assert get_tuning("drop_d")[5] == 38          # low string dropped to D
    assert get_tuning("half_step_down")[0] == 63  # Eb
