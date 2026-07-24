"""Evaluation metrics.

  - Note F1        : a match requires pitch exact and onset within onset_tol_ms
  - Articulation   : P/R/F1 per class, over correctly matched notes only
  - String/fret    : % of matched notes with the correct (string, fret) pair
  - Playability    : % of consecutive transitions needing a hand shift > N frets

Pure logic — operates on NoteEvent/TabNote lists, no audio required.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Articulation, NoteEvent, TabNote

# techniques we score, drawn from entry/exit/interior annotations
_TECH_CLASSES = [
    Articulation.HAMMER_ON, Articulation.PULL_OFF, Articulation.SLIDE_IN,
    Articulation.SLIDE_OUT, Articulation.BEND, Articulation.RELEASE,
    Articulation.VIBRATO,
]


def note_labels(n: NoteEvent) -> set[str]:
    """Multi-label technique set for a note (entry + exit + vibrato)."""
    out: set[str] = set()
    if n.entry in (Articulation.HAMMER_ON, Articulation.PULL_OFF, Articulation.SLIDE_IN):
        out.add(n.entry.value)
    if n.exit in (Articulation.SLIDE_OUT, Articulation.BEND, Articulation.RELEASE):
        out.add(n.exit.value)
    if n.bend is not None:
        out.add(Articulation.BEND.value if not n.bend.released else Articulation.RELEASE.value)
    if n.vibrato_hz is not None:
        out.add(Articulation.VIBRATO.value)
    return out


def match_notes(pred: list[NoteEvent], gt: list[NoteEvent],
                onset_tol_s: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching: same midi and onset within tolerance."""
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi, g in enumerate(gt):
        best, best_dt = -1, onset_tol_s + 1e-9
        for pi, p in enumerate(pred):
            if pi in used_pred or p.midi != g.midi:
                continue
            dt = abs(p.start_s - g.start_s)
            if dt <= onset_tol_s and dt < best_dt:
                best, best_dt = pi, dt
        if best >= 0:
            used_pred.add(best)
            pairs.append((best, gi))
    return pairs


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(p, r, f, tp, fp, fn)


def note_f1(pred: list[NoteEvent], gt: list[NoteEvent], onset_tol_s: float) -> PRF:
    pairs = match_notes(pred, gt, onset_tol_s)
    tp = len(pairs)
    return _prf(tp, len(pred) - tp, len(gt) - tp)


def articulation_prf(pred: list[NoteEvent], gt: list[NoteEvent],
                     onset_tol_s: float) -> dict[str, PRF]:
    """Per-class P/R/F1 over correctly matched notes only."""
    pairs = match_notes(pred, gt, onset_tol_s)
    result: dict[str, PRF] = {}
    for cls in _TECH_CLASSES:
        c = cls.value
        tp = fp = fn = 0
        for pi, gi in pairs:
            in_pred = c in note_labels(pred[pi])
            in_gt = c in note_labels(gt[gi])
            if in_pred and in_gt:
                tp += 1
            elif in_pred and not in_gt:
                fp += 1
            elif in_gt and not in_pred:
                fn += 1
        if tp + fp + fn > 0:
            result[c] = _prf(tp, fp, fn)
    return result


def string_fret_accuracy(pred: list[TabNote], gt: list[TabNote],
                         onset_tol_s: float) -> float:
    pairs = match_notes(pred, gt, onset_tol_s)
    scored = correct = 0
    for pi, gi in pairs:
        g = gt[gi]
        if not isinstance(g, TabNote) or g.string < 0:
            continue
        scored += 1
        p = pred[pi]
        if isinstance(p, TabNote) and p.string == g.string and p.fret == g.fret:
            correct += 1
    return correct / scored if scored else 0.0


def playability(pred: list[TabNote], shift_frets: int) -> float:
    """Fraction of consecutive transitions needing a hand shift > shift_frets."""
    tabs = [n for n in sorted(pred, key=lambda x: x.start_s)
            if isinstance(n, TabNote) and n.fret >= 0]
    if len(tabs) < 2:
        return 0.0
    bad = sum(1 for a, b in zip(tabs[:-1], tabs[1:])
              if abs(b.fret - a.fret) > shift_frets)
    return bad / (len(tabs) - 1)


def evaluate(pred: list[NoteEvent], gt: list[NoteEvent],
             onset_tol_ms: float = 50.0, shift_frets: int = 5) -> dict:
    tol = onset_tol_ms / 1000.0
    nf1 = note_f1(pred, gt, tol)
    art = articulation_prf(pred, gt, tol)
    art_f1_agg = (sum(p.f1 for p in art.values()) / len(art)) if art else 0.0
    return {
        "note_f1": nf1.f1,
        "note_precision": nf1.precision,
        "note_recall": nf1.recall,
        "articulation_f1_macro": art_f1_agg,
        "articulation_per_class": {k: v.f1 for k, v in art.items()},
        "string_fret_accuracy": string_fret_accuracy(pred, gt, tol),
        "playability_bad_shift_frac": playability(pred, shift_frets),
        "n_pred": len(pred),
        "n_gt": len(gt),
    }
