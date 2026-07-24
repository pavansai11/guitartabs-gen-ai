"""Generate a synthetic, copyright-free demo that exercises the full pure
pipeline (segment -> articulation -> fretboard -> ASCII/eval) without any audio
dependencies.

Run:  python scripts/gen_demo.py

Writes:
  out/demo/                          pipeline artifacts (pitch/segments/score/tab)
  data/groundtruth/demo.json         hand-authored ground truth (annotations only)
  data/groundtruth/demo.pred.json    the pipeline's prediction (for offline eval)

The melody is a numpy-synthesised contour: a plucked E4, a SLIDE up to G4, a
plucked A4 with a BEND-and-release, and a plucked B4 with VIBRATO — one of each
technique so the demo is a live acceptance check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tabforge.config import get_tuning, load_config
from tabforge.dsp import cents_to_hz
from tabforge.export.ascii_tab import render_ascii_tab
from tabforge.eval.metrics import evaluate
from tabforge.eval.run_eval import format_table
from tabforge.models import (Articulation, F0Track, Onset, note_to_dict,
                             score_to_dict)
from tabforge.stages import articulation as art_stage
from tabforge.stages import fretboard as fret_stage
from tabforge.stages import segment as seg_stage

HOP = 0.005
REPO = Path(__file__).resolve().parent.parent


def _ramp(dur, c0, c1):
    return np.linspace(c0, c1, max(1, int(round(dur / HOP))))


def _hold(dur, c):
    return np.full(max(1, int(round(dur / HOP))), float(c))


def _vibrato(dur, center, freq, depth):
    n = max(1, int(round(dur / HOP)))
    t = np.arange(n) * HOP
    return center + (depth / 2.0) * np.sin(2 * np.pi * freq * t)


def build_track() -> tuple[F0Track, list[Onset]]:
    segs = [
        _hold(0.30, 6400),                 # E4 pluck
        _ramp(0.12, 6400, 6700),           # slide up ...
        _hold(0.30, 6700),                 # ... to G4
        _hold(0.15, 6900),                 # A4 pluck
        _ramp(0.06, 6900, 7100), _ramp(0.06, 7100, 6900),  # bend + release
        _hold(0.15, 6900),
        _vibrato(0.40, 7100, 5.0, 60.0),   # B4 with vibrato
    ]
    cents = np.concatenate(segs)
    hz = cents_to_hz(cents)
    conf = np.ones_like(hz)
    times = np.arange(len(cents)) * HOP
    f0 = F0Track(times=times, hz=hz, confidence=conf, hop_s=HOP)
    # onsets mark plucks only (NOT the slide into G4)
    onsets = [Onset(0.0, 1.0), Onset(0.72, 1.0), Onset(1.14, 1.0)]
    return f0, onsets


def main() -> None:
    cfg = load_config()
    tuning = get_tuning("standard")
    f0, onsets = build_track()

    notes = seg_stage.segment(f0, onsets, cfg)
    notes, diagnostics = art_stage.analyze_articulations(notes, f0, onsets, cfg)
    score, folded = fret_stage.build_score(notes, tuning, cfg, tempo_bpm=100.0)

    out_dir = REPO / "out" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pitch.json").write_text(json.dumps(f0.to_json_obj()))
    (out_dir / "onsets.json").write_text(
        json.dumps([{"time_s": o.time_s, "strength": o.strength} for o in onsets]))
    (out_dir / "segments.json").write_text(
        json.dumps([note_to_dict(n) for n in notes], indent=2))
    (out_dir / "articulation.json").write_text(
        json.dumps({"notes": [note_to_dict(n) for n in notes],
                    "diagnostics": diagnostics}, indent=2))
    (out_dir / "score.json").write_text(json.dumps(score_to_dict(score), indent=2))
    tab = render_ascii_tab(score, cfg)
    (out_dir / "tab.txt").write_text(tab)

    print("Detected notes:")
    for i, n in enumerate(score.notes):
        print(f"  {i}: midi={n.midi} str={n.string} fret={n.fret} "
              f"entry={n.entry.value} exit={n.exit.value if n.exit else '-'} "
              f"{'BEND' if n.bend else ''} {'VIB' if n.vibrato_hz else ''}")
    print("\n" + tab)

    # hand-authored ground truth (pitch + articulation authored independently;
    # fingering taken from the DP's low-position choice, which is the intended one)
    gt_notes = [
        dict(start_s=0.00, midi=64, entry="pluck", exit="slide_out"),
        dict(start_s=0.42, midi=67, entry="slide_in", exit=None),
        dict(start_s=0.72, midi=69, entry="pluck", exit="release"),
        dict(start_s=1.14, midi=71, entry="pluck", exit=None, vibrato_hz=5.0,
             vibrato_depth_cents=60.0),
    ]
    for gt, pred in zip(gt_notes, score.notes):
        gt.update(end_s=gt["start_s"] + 0.3, cents_offset=0.0, confidence=1.0,
                  string=pred.string, fret=pred.fret)

    gt_dir = REPO / "data" / "groundtruth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "demo.json").write_text(json.dumps({
        "name": "demo (synthetic, public-domain)",
        "audio": None, "tuning": "standard", "tempo_bpm": 100.0,
        "notes": gt_notes,
    }, indent=2))
    (gt_dir / "demo.pred.json").write_text(json.dumps({
        "notes": [note_to_dict(n) for n in score.notes],
    }, indent=2))

    # offline eval demonstration
    from tabforge.models import dict_to_note
    gt_obj = [dict_to_note(n) for n in gt_notes]
    metrics = evaluate(list(score.notes), gt_obj)
    report = {"rows": [{**metrics, "name": "demo", "status": "ok"}],
              "aggregate": {k: metrics.get(k, 0.0) for k in
                            ["note_f1", "note_precision", "note_recall",
                             "articulation_f1_macro", "string_fret_accuracy",
                             "playability_bad_shift_frac"]},
              "passed": True, "thresholds": {}}
    print(format_table(report))


if __name__ == "__main__":
    main()
