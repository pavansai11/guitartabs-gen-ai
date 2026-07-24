"""ASCII export (Stage 9) and evaluation metrics (Stage 8 harness)."""

from __future__ import annotations

from tabforge.config import get_tuning, load_config
from tabforge.eval.metrics import (evaluate, note_f1, playability,
                                    string_fret_accuracy)
from tabforge.export.ascii_tab import render_ascii_tab
from tabforge.models import Articulation, BendCurve, Score, TabNote
from synth import note


def _tab(midi, string, fret, **kw):
    return TabNote(start_s=kw.pop("start_s", 0.0), end_s=kw.pop("end_s", 0.25),
                   midi=midi, cents_offset=0.0, confidence=1.0,
                   string=string, fret=fret, **kw)


def test_ascii_tab_has_six_lines_and_symbols():
    notes = [
        _tab(64, 0, 0, start_s=0.0, end_s=0.25),
        _tab(67, 0, 3, start_s=0.25, end_s=0.5, entry=Articulation.SLIDE_IN),
        _tab(69, 0, 5, start_s=0.5, end_s=0.75, entry=Articulation.HAMMER_ON),
        _tab(69, 1, 7, start_s=0.75, end_s=1.0, vibrato_hz=5.0),
    ]
    sc = Score(notes=notes, tempo_bpm=120.0, time_signature=(4, 4),
               tuning=get_tuning("standard"))
    tab = render_ascii_tab(sc, load_config())
    body = [ln for ln in tab.splitlines() if ln and not ln.startswith("#")]
    assert len(body) >= 6
    assert "/" in tab                 # slide symbol
    assert "h" in tab                 # hammer symbol
    assert "~" in tab                 # vibrato symbol


def test_note_f1_perfect_match():
    gt = [note(0.0, 0.25, 64), note(0.3, 0.55, 67)]
    pred = [note(0.01, 0.25, 64), note(0.31, 0.55, 67)]  # within 50 ms
    prf = note_f1(pred, gt, onset_tol_s=0.05)
    assert prf.f1 == 1.0


def test_note_f1_penalises_wrong_pitch():
    gt = [note(0.0, 0.25, 64)]
    pred = [note(0.0, 0.25, 65)]      # wrong pitch -> no match
    prf = note_f1(pred, gt, onset_tol_s=0.05)
    assert prf.f1 == 0.0


def test_string_fret_accuracy():
    gt = [_tab(64, 0, 0, start_s=0.0), _tab(67, 0, 3, start_s=0.3)]
    pred = [_tab(64, 0, 0, start_s=0.0), _tab(67, 1, 8, start_s=0.3)]
    acc = string_fret_accuracy(pred, gt, onset_tol_s=0.05)
    assert acc == 0.5                 # one of two correct


def test_playability_flags_big_shift():
    tabs = [_tab(64, 0, 1, start_s=0.0), _tab(88, 0, 15, start_s=0.3)]
    frac = playability(tabs, shift_frets=5)
    assert frac == 1.0                # 14-fret jump > 5


def test_evaluate_returns_all_metrics():
    gt = [note(0.0, 0.25, 64, exit=Articulation.SLIDE_OUT),
          note(0.3, 0.55, 67, entry=Articulation.SLIDE_IN)]
    pred = [note(0.0, 0.25, 64, exit=Articulation.SLIDE_OUT),
            note(0.3, 0.55, 67, entry=Articulation.SLIDE_IN)]
    m = evaluate(pred, gt)
    assert m["note_f1"] == 1.0
    assert m["articulation_f1_macro"] == 1.0
