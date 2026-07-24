"""Review-UI correction logic (Stage 10) — the pure parts, no FastAPI needed."""

from __future__ import annotations

import json

from tabforge.config import get_tuning
from tabforge.models import Articulation, Score, TabNote, score_to_dict
from tabforge.review.app import (append_correction, apply_corrections,
                                 corrected_score, load_corrections)


def _demo_score():
    notes = [
        TabNote(start_s=0.0, end_s=0.3, midi=64, cents_offset=0.0, confidence=1.0,
                string=0, fret=0),
        TabNote(start_s=0.3, end_s=0.6, midi=67, cents_offset=0.0, confidence=1.0,
                string=0, fret=3),
    ]
    return Score(notes=notes, tempo_bpm=100.0, time_signature=(4, 4),
                 tuning=get_tuning("standard"))


def test_apply_corrections_edits_note():
    sc = _demo_score()
    corrected = apply_corrections(sc, [
        {"note_index": 1, "fret": 8, "string": 1, "entry": "hammer_on"},
    ])
    assert corrected.notes[1].fret == 8
    assert corrected.notes[1].string == 1
    assert corrected.notes[1].entry == Articulation.HAMMER_ON
    # original untouched (frozen dataclasses)
    assert sc.notes[1].fret == 3


def test_corrections_are_appended_never_overwritten(tmp_path):
    (tmp_path / "score.json").write_text(json.dumps(score_to_dict(_demo_score())))
    append_correction(tmp_path, {"note_index": 0, "fret": 2})
    append_correction(tmp_path, {"note_index": 1, "fret": 5})
    saved = load_corrections(tmp_path)
    assert len(saved) == 2                      # both retained
    assert all("ts" in c for c in saved)        # timestamped

    corrected = corrected_score(tmp_path)
    assert corrected.notes[0].fret == 2
    assert corrected.notes[1].fret == 5
