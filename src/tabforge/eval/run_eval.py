"""Evaluation runner.

Loads ground-truth JSON (checked in — annotations only, never audio), obtains
predictions, computes metrics, and prints a per-excerpt and aggregate table.
Regressions past the configured thresholds return a non-zero status so CI can
fail on them.

Predictions are obtained by:
  1. running the pipeline on data/samples/<audio> when the audio is present, or
  2. loading a sibling ``<name>.pred.json`` (useful for tests / offline CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..models import NoteEvent, dict_to_note
from .metrics import evaluate

# Minimum acceptable aggregate metrics (M2/M3/M4 acceptance criteria).
DEFAULT_THRESHOLDS = {
    "note_f1": 0.70,
    "articulation_f1_macro": 0.60,
    "string_fret_accuracy": 0.80,
}


def load_notes(items: list[dict[str, Any]]) -> list[NoteEvent]:
    return [dict_to_note(x) for x in items]


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["_path"] = str(path)
    data["notes_obj"] = load_notes(data.get("notes", []))
    return data


def _predictions_for(gt: dict[str, Any], samples_dir: Path, out_root: Path,
                     cfg: Config) -> list[NoteEvent] | None:
    # 1. sibling prediction file (offline path)
    gt_path = Path(gt["_path"])
    pred_path = gt_path.with_suffix(".pred.json")
    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as fh:
            return load_notes(json.load(fh).get("notes", []))

    # 2. run the pipeline on the referenced audio, if present
    audio_name = gt.get("audio")
    if audio_name:
        audio_path = samples_dir / audio_name
        if audio_path.exists():
            from ..pipeline import transcribe  # lazy: pulls audio stack
            result = transcribe(
                audio_path, cfg=cfg, out_root=out_root,
                source=gt.get("source", "auto"),
                tuning_name=gt.get("tuning", "standard"),
                capo=int(gt.get("capo", 0)),
                quantize=gt.get("quantize", True),
            )
            return list(result.score.notes)
    return None


def run_eval(gt_dir: str | Path = "data/groundtruth",
             samples_dir: str | Path = "data/samples",
             out_root: str | Path = "out",
             preset: str | None = None,
             thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    cfg = load_config(preset=preset)
    gt_dir = Path(gt_dir)
    samples_dir = Path(samples_dir)
    out_root = Path(out_root)
    thresholds = thresholds or DEFAULT_THRESHOLDS

    gt_files = sorted(gt_dir.glob("*.json"))
    gt_files = [p for p in gt_files if not p.name.endswith(".pred.json")]

    rows: list[dict[str, Any]] = []
    onset_tol = float(cfg.get("eval", "onset_tol_ms", 50.0))
    shift = int(cfg.get("eval", "playability_shift_frets", 5))

    for gt_path in gt_files:
        gt = load_ground_truth(gt_path)
        pred = _predictions_for(gt, samples_dir, out_root, cfg)
        if pred is None:
            rows.append({"name": gt.get("name", gt_path.stem), "status": "SKIP (no prediction/audio)"})
            continue
        m = evaluate(pred, gt["notes_obj"], onset_tol_ms=onset_tol, shift_frets=shift)
        m["name"] = gt.get("name", gt_path.stem)
        m["status"] = "ok"
        rows.append(m)

    scored = [r for r in rows if r.get("status") == "ok"]
    agg = _aggregate(scored)
    report = {"rows": rows, "aggregate": agg,
              "passed": _check_thresholds(agg, thresholds) if scored else False,
              "thresholds": thresholds}
    return report


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ["note_f1", "note_precision", "note_recall", "articulation_f1_macro",
            "string_fret_accuracy", "playability_bad_shift_frac"]
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}


def _check_thresholds(agg: dict[str, float], thresholds: dict[str, float]) -> bool:
    return all(agg.get(k, 0.0) >= v for k, v in thresholds.items())


def format_table(report: dict[str, Any]) -> str:
    lines = []
    header = f"{'excerpt':<24}{'noteF1':>8}{'artF1':>8}{'str/fret':>10}{'play>5':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in report["rows"]:
        if r.get("status") != "ok":
            lines.append(f"{r['name']:<24}{r.get('status', 'SKIP'):>34}")
            continue
        lines.append(
            f"{r['name']:<24}{r['note_f1']:>8.3f}{r['articulation_f1_macro']:>8.3f}"
            f"{r['string_fret_accuracy']:>10.3f}{r['playability_bad_shift_frac']:>8.3f}"
        )
    agg = report.get("aggregate", {})
    if agg:
        lines.append("-" * len(header))
        lines.append(
            f"{'AGGREGATE':<24}{agg['note_f1']:>8.3f}{agg['articulation_f1_macro']:>8.3f}"
            f"{agg['string_fret_accuracy']:>10.3f}{agg['playability_bad_shift_frac']:>8.3f}"
        )
        status = "PASS" if report.get("passed") else "FAIL"
        lines.append(f"thresholds: {report['thresholds']}  ->  {status}")
    return "\n".join(lines)
