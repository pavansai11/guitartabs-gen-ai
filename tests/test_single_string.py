"""Single-string vocal-melody fingering (the "Kanmani Anbodu" repertoire).

Reference tabs for this repertoire put the whole sung melody on the B string
with slides. These tests assert the single_string strategy reproduces that.
"""

from __future__ import annotations

from tabforge.config import get_tuning, load_config
from tabforge.models import Articulation
from tabforge.stages.fretboard import assign_single_string, build_score
from synth import note

STD = get_tuning("standard")


def test_kanmani_line_lands_on_b_string_with_reference_frets():
    cfg = load_config(preset="indian")
    # actual pitches of "Kanmani Anbodu Kaadhalan" line 1 (B string frets 5 6 10
    # 8 3 5 8 4 -> MIDI with B open = 59)
    midis = [64, 65, 69, 67, 62, 64, 67, 63]
    notes = [note(i * 0.3, i * 0.3 + 0.25, m) for i, m in enumerate(midis)]
    tabs = assign_single_string(notes, STD, string_idx=1, cfg=cfg)
    assert all(t.string == 1 for t in tabs)             # all on the B string
    assert [t.fret for t in tabs] == [5, 6, 10, 8, 3, 5, 8, 4]


def test_single_string_octave_normalises_wide_leaps():
    cfg = load_config()
    # a note an octave above the previous should fold to the nearest fret, not
    # jump 12 frets away
    notes = [note(0.0, 0.2, 64), note(0.3, 0.5, 76)]   # E4 then E5
    tabs = assign_single_string(notes, STD, string_idx=1, cfg=cfg)
    assert tabs[0].string == tabs[1].string == 1
    assert abs(tabs[1].fret - tabs[0].fret) <= 2        # stays near, no big shift


def test_indian_preset_selects_single_string_strategy():
    cfg = load_config(preset="indian")
    assert cfg.get("fretboard", "strategy") == "single_string"
    assert cfg.get("fretboard", "melody_string") == 1


def test_build_score_dp_still_default_without_preset():
    cfg = load_config()
    assert cfg.get("fretboard", "strategy") == "dp"
    notes = [note(0.0, 0.2, 64, exit=Articulation.SLIDE_OUT),
             note(0.3, 0.5, 67, entry=Articulation.SLIDE_IN)]
    score, _ = build_score(notes, STD, cfg)
    # DP still honours the same-string slide constraint
    assert score.notes[0].string == score.notes[1].string
