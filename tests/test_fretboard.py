"""Fretboard DP assignment (Stage 8): constraints and cost behaviour."""

from __future__ import annotations

from tabforge.config import get_tuning, load_config
from tabforge.models import Articulation, BendCurve
from tabforge.stages.fretboard import assign_fretboard, candidates
from synth import note

STD = None


def _std():
    return get_tuning("standard")


def test_candidates_for_open_high_e():
    # MIDI 64 = high E: open string 0 fret 0, plus higher positions
    cands = candidates(64, _std(), capo=0, max_fret=17)
    assert (0, 0) in cands
    assert (1, 5) in cands           # B string, 5th fret = 64


def test_slide_forces_same_string():
    cfg = load_config()
    a = note(0.0, 0.3, 64, exit=Articulation.SLIDE_OUT)
    b = note(0.3, 0.6, 67, entry=Articulation.SLIDE_IN)
    tabs, _ = assign_fretboard([a, b], _std(), cfg)
    assert tabs[0].string == tabs[1].string   # hard constraint


def test_hammer_forces_same_string():
    cfg = load_config()
    a = note(0.0, 0.3, 64)
    b = note(0.3, 0.6, 67, entry=Articulation.HAMMER_ON)
    tabs, _ = assign_fretboard([a, b], _std(), cfg)
    assert tabs[0].string == tabs[1].string


def test_bend_cannot_be_open_string():
    cfg = load_config()
    # MIDI 64 is playable open (string 0, fret 0); a bend must pick a fretted spot
    n = note(0.0, 0.5, 64,
             bend=BendCurve(peak_cents=200.0, onset_s=0.1, hold_s=0.1, released=False),
             exit=Articulation.BEND)
    tabs, _ = assign_fretboard([n], _std(), cfg)
    assert tabs[0].fret >= 1


def test_low_position_preferred_for_simple_line():
    cfg = load_config()
    line = [note(i * 0.3, i * 0.3 + 0.25, 64 + i) for i in range(4)]
    tabs, _ = assign_fretboard(line, _std(), cfg)
    # nothing forces high positions; the DP should keep frets low
    assert max(t.fret for t in tabs) <= 12


def test_octave_folding_for_out_of_range_note():
    cfg = load_config()
    # MIDI 30 is below the guitar's lowest open string (40) — must be folded up
    n = note(0.0, 0.5, 30)
    tabs, folded = assign_fretboard([n], _std(), cfg)
    assert 0 in folded
    assert tabs[0].fret >= 0
