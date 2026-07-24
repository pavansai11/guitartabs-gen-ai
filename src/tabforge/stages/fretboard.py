"""Stage 8 — Fretboard assignment.

The same pitch is playable in up to six places. Choose positions by dynamic
programming over the note sequence (Sayegh's optimum-path formulation).

Transition cost between consecutive candidates:
    cost = w_pos  * |fret_n - fret_prev|      # hand movement
         + w_str  * |string_n - string_prev|  # string crossing
         + w_high * max(0, fret_n - 12)        # prefer lower positions
         - w_open * (1 if fret_n == 0 else 0)  # open strings are cheap
         + INF if articulation constraint violated

Articulation constraints are HARD, not costs — this is why articulation
detection must run BEFORE fret assignment:
    SLIDE_IN/OUT, HAMMER_ON, PULL_OFF  -> both notes on the SAME string
    BEND                               -> fret >= 1 (cannot bend an open
                                          string); on the top three strings,
                                          prefer frets >= comfort_min_fret

Pure numpy/python: runs and is tested without any audio stack.
"""

from __future__ import annotations

from dataclasses import replace

from ..config import Config
from ..models import Articulation, NoteEvent, Score, TabNote

INF = float("inf")

_SAME_STRING_ENTRY = {Articulation.SLIDE_IN, Articulation.HAMMER_ON,
                      Articulation.PULL_OFF}
_SAME_STRING_EXIT = {Articulation.SLIDE_OUT}


def _is_bend(note: NoteEvent) -> bool:
    return note.bend is not None or note.exit in (Articulation.BEND, Articulation.RELEASE)


def candidates(midi: int, tuning: list[int], capo: int, max_fret: int,
               require_fretted: bool = False) -> list[tuple[int, int]]:
    """(string, fret) pairs producing ``midi`` in the given tuning/capo."""
    out = []
    for s, open_midi in enumerate(tuning):
        fret = midi - open_midi - capo
        lo = 1 if require_fretted else 0
        if lo <= fret <= max_fret:
            out.append((s, fret))
    return out


def _fold_into_range(midi: int, tuning: list[int], capo: int, max_fret: int
                     ) -> tuple[int, bool]:
    """Octave-fold a note into playable range (Known Hard Parts: vocal melodies
    exceed comfortable guitar range). Returns (midi, folded)."""
    if candidates(midi, tuning, capo, max_fret):
        return midi, False
    for shift in (12, -12, 24, -24, 36, -36):
        if candidates(midi + shift, tuning, capo, max_fret):
            return midi + shift, True
    return midi, False


def _unary_cost(string: int, fret: int, note: NoteEvent, cfg: Config) -> float:
    w_high = float(cfg.get("fretboard", "w_high", 0.5))
    w_open = float(cfg.get("fretboard", "w_open", 1.5))
    comfort = int(cfg.get("fretboard", "bend_comfort_min_fret", 5))
    cost = w_high * max(0, fret - 12) - w_open * (1.0 if fret == 0 else 0.0)
    # soft bend-comfort preference on the top three strings
    if _is_bend(note) and string <= 2 and fret < comfort:
        cost += 1.0
    return cost


def _pairwise_cost(prev: tuple[int, int], cur: tuple[int, int],
                   a: NoteEvent, b: NoteEvent, cfg: Config) -> float:
    w_pos = float(cfg.get("fretboard", "w_pos", 1.0))
    w_str = float(cfg.get("fretboard", "w_str", 0.7))
    ps, pf = prev
    cs, cf = cur
    same_string_required = (b.entry in _SAME_STRING_ENTRY) or (a.exit in _SAME_STRING_EXIT)
    if same_string_required and cs != ps:
        return INF
    return w_pos * abs(cf - pf) + w_str * abs(cs - ps)


def assign_fretboard(notes: list[NoteEvent], tuning: list[int], cfg: Config,
                     capo: int | None = None) -> tuple[list[TabNote], list[int]]:
    """DP assignment. Returns (tab_notes, indices_of_octave_folded_notes)."""
    max_fret = int(cfg.get("fretboard", "max_fret", 17))
    if capo is None:
        capo = int(cfg.get("fretboard", "capo", 0))

    # candidate generation (with octave folding + bend fret>=1 constraint)
    cand: list[list[tuple[int, int]]] = []
    folded: list[int] = []
    play_notes: list[NoteEvent] = []
    for i, n in enumerate(notes):
        midi, was_folded = _fold_into_range(n.midi, tuning, capo, max_fret)
        if was_folded:
            folded.append(i)
            n = replace(n, midi=midi)
        play_notes.append(n)
        c = candidates(midi, tuning, capo, max_fret,
                       require_fretted=_is_bend(n))
        if not c:  # unplayable even after folding — fall back to any candidate
            c = candidates(midi, tuning, capo, max_fret) or [(0, 0)]
        cand.append(c)

    n_notes = len(play_notes)
    if n_notes == 0:
        return [], folded

    # Viterbi over candidates
    dp: list[list[float]] = [[0.0] * len(cs) for cs in cand]
    back: list[list[int]] = [[-1] * len(cs) for cs in cand]
    for k, (s, f) in enumerate(cand[0]):
        dp[0][k] = _unary_cost(s, f, play_notes[0], cfg)

    for i in range(1, n_notes):
        for k, cur in enumerate(cand[i]):
            best, best_j = INF, -1
            u = _unary_cost(cur[0], cur[1], play_notes[i], cfg)
            for j, prev in enumerate(cand[i - 1]):
                if dp[i - 1][j] == INF:
                    continue
                trans = _pairwise_cost(prev, cur, play_notes[i - 1],
                                       play_notes[i], cfg)
                total = dp[i - 1][j] + trans + u
                if total < best:
                    best, best_j = total, j
            dp[i][k] = best
            back[i][k] = best_j

    # backtrack from the best final candidate
    last = min(range(len(cand[-1])), key=lambda k: dp[-1][k])
    path = [0] * n_notes
    path[-1] = last
    for i in range(n_notes - 1, 0, -1):
        prev = back[i][path[i]]
        path[i - 1] = prev if prev >= 0 else 0

    tab_notes = []
    for i, n in enumerate(play_notes):
        s, f = cand[i][path[i]]
        tab_notes.append(TabNote.from_note(n, string=s, fret=f))
    return tab_notes, folded


def build_score(notes: list[NoteEvent], tuning: list[int], cfg: Config,
                tempo_bpm: float = 120.0,
                time_signature: tuple[int, int] = (4, 4),
                capo: int | None = None) -> tuple[Score, list[int]]:
    if capo is None:
        capo = int(cfg.get("fretboard", "capo", 0))
    tab_notes, folded = assign_fretboard(notes, tuning, cfg, capo=capo)
    score = Score(notes=tab_notes, tempo_bpm=tempo_bpm,
                  time_signature=time_signature, tuning=tuning, capo=capo)
    return score, folded
