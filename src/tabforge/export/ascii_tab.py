"""ASCII tab export.

Six lines, articulation symbols:
    /  slide up      \\  slide down
    h  hammer-on     p  pull-off
    b  bend          r  release
    ~  vibrato
Wrapped at a configurable width with bar lines derived from tempo/time sig.
Pure logic — no audio or notation dependencies.
"""

from __future__ import annotations

from ..config import Config
from ..models import Articulation, Score, TabNote

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _string_labels(tuning: list[int]) -> list[str]:
    labels = [_NOTE_NAMES[m % 12] for m in tuning]
    width = max(len(x) for x in labels)
    return [x.ljust(width) for x in labels]


def _prefix_symbol(note: TabNote, prev: TabNote | None) -> str:
    if note.entry == Articulation.SLIDE_IN:
        if prev is not None:
            return "/" if note.midi >= prev.midi else "\\"
        return "/"
    if note.entry == Articulation.HAMMER_ON:
        return "h"
    if note.entry == Articulation.PULL_OFF:
        return "p"
    return ""


def _suffix_symbol(note: TabNote) -> str:
    out = ""
    if note.exit == Articulation.RELEASE or (note.bend is not None and note.bend.released):
        out += "br"
    elif note.exit == Articulation.BEND or note.bend is not None:
        out += "b"
    if note.vibrato_hz is not None:
        out += "~"
    return out


def _measure_length_s(score: Score) -> float:
    beats_per_measure = score.time_signature[0]
    if score.tempo_bpm <= 0:
        return float("inf")
    return beats_per_measure * (60.0 / score.tempo_bpm)


def render_ascii_tab(score: Score, cfg: Config | None = None,
                     wrap: int | None = None) -> str:
    """Render a Score to wrapped six-line ASCII tablature."""
    if wrap is None:
        wrap = int(cfg.get("export", "ascii_wrap", 80)) if cfg else 80
    n_str = len(score.tuning)
    labels = _string_labels(score.tuning)
    label_w = len(labels[0])

    # Build columns. Each column is a list of n_str equal-width cells.
    columns: list[list[str]] = []

    def add_bar() -> None:
        columns.append(["|"] * n_str)

    def add_spacer() -> None:
        columns.append(["-"] * n_str)

    measure_len = _measure_length_s(score)
    cur_measure = 0
    add_bar()
    prev: TabNote | None = None
    ordered = sorted(score.notes, key=lambda x: x.start_s)
    for note in ordered:
        if measure_len != float("inf"):
            m = int(note.start_s // measure_len)
            if m > cur_measure:
                for _ in range(m - cur_measure):
                    add_bar()
                cur_measure = m

        token = f"{_prefix_symbol(note, prev)}{note.fret}{_suffix_symbol(note)}"
        width = len(token)
        col = []
        for s in range(n_str):
            if s == note.string:
                col.append(token)
            else:
                col.append("-" * width)
        columns.append(col)
        add_spacer()
        prev = note
    add_bar()

    # Assemble into wrapped blocks.
    budget = max(20, wrap - label_w - 1)
    blocks: list[list[str]] = []
    cur_cols: list[list[str]] = []
    cur_width = 0
    for col in columns:
        w = len(col[0])
        if cur_cols and cur_width + w > budget:
            blocks.append(cur_cols)
            cur_cols = []
            cur_width = 0
        cur_cols.append(col)
        cur_width += w
    if cur_cols:
        blocks.append(cur_cols)

    lines: list[str] = []
    for block in blocks:
        for s in range(n_str):
            row = "".join(col[s] for col in block)
            lines.append(f"{labels[s]}{row}")
        lines.append("")  # blank line between blocks
    header = _legend()
    return header + "\n".join(lines).rstrip() + "\n"


def _legend() -> str:
    return (
        "# TabForge ASCII tab   "
        "/ slide up  \\ slide down  h hammer  p pull  b bend  r release  ~ vibrato\n"
    )
