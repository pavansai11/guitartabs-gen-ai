"""Stage 6 — Articulation analysis. ⭐ THE differentiating module.

Runs on the continuous cents contour (rule 1: the pitch contour is the product)
BEFORE any quantisation. All thresholds live in config/default.yaml and are
overlaid by presets (e.g. presets/indian.yaml widens the slide window for
meend) — tunable without code changes.

Classification of a transition A(p1) -> B(p2), tested in this order and
stopping at the first match (rule: a slide misread as a bend is the most common
failure; the discriminator is that a slide establishes a NEW stable semitone
band at its destination and a bend does not):

  1. pluck        onset within +/- pluck_window_ms of B's start
  2. hammer/pull  no onset; dur < legato_max_ms; |p2-p1| >= interval;
                  discontinuous (max frame delta > legato_discontinuity_cents)
  3. slide        no onset; slide_min_ms <= dur <= slide_max_ms; monotonic;
                  |p2-p1| >= interval; R2 of linear fit >= slide_r2_min
  4. bend         (per-note) contour rises from stable pitch by 80-300 cents,
                  no new stable band, then holds/returns; released if it comes
                  back within bend_release_tol_cents
  5. vibrato      (per-note, dur >= vibrato_min_ms) dominant FFT freq in
                  [4,9] Hz and peak-to-peak depth in [20,120] cents

Emits a per-note diagnostics dict (slope, R2, transition duration, onset
distance, ...) for debugging and the review UI.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from ..config import Config
from ..dsp import linfit_r2, longest_stable_run
from ..models import Articulation, BendCurve, NoteEvent, Onset


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _nominal_cents(note: NoteEvent) -> float:
    return note.midi * 100.0 + note.cents_offset


def _frame(t: float, hop_s: float) -> int:
    return int(round(t / hop_s))


def _slice(cents: np.ndarray, times: np.ndarray, t0: float, t1: float,
           hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    # Half-open [t0, t1): a note spanning frames [a, b) has end_s == times[b],
    # so an inclusive slice would leak the next note's first frame into this
    # note's interior (and misread the pitch jump as a bend).
    a = max(0, _frame(t0, hop_s))
    b = min(len(cents), _frame(t1, hop_s))
    return cents[a:b], times[a:b]


def _drop_nan(c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = ~np.isnan(c)
    return c[m], t[m]


def _glide_span(c: np.ndarray, p1: float, p2: float,
                band_tol: float) -> tuple[int, int] | None:
    """Indices (i0, i1) of the glide from the last frame near p1 to the first
    frame near p2 after it. None if the contour never reaches both bands."""
    near1 = np.abs(c - p1) <= band_tol
    near2 = np.abs(c - p2) <= band_tol
    if not near1.any() or not near2.any():
        return None
    last1 = int(np.max(np.where(near1)[0]))
    after = np.where(near2)[0]
    after = after[after > last1]
    if after.size == 0:
        return None
    first2 = int(after.min())
    return last1, first2


def _monotonic_frac(c: np.ndarray, direction: float) -> float:
    """Fraction of non-negligible steps that follow the glide direction."""
    d = np.diff(c)
    signif = np.abs(d) > 5.0
    if not signif.any():
        return 1.0
    return float(np.mean(np.sign(d[signif]) == np.sign(direction)))


# --------------------------------------------------------------------------- #
# transition classification (pluck / hammer / pull / slide)
# --------------------------------------------------------------------------- #

def classify_transition(a: NoteEvent, b: NoteEvent, cents: np.ndarray,
                        times: np.ndarray, hop_s: float,
                        onsets: list[Onset], cfg: Config) -> dict[str, Any]:
    art = cfg.section("articulation")
    pluck_win = float(cfg.get("onsets", "pluck_window_ms", 30.0)) / 1000.0
    pad = float(art.get("transition_pad_ms", 60.0)) / 1000.0
    band_tol = float(cfg.get("segment", "new_band_cents", 40.0))

    p1 = _nominal_cents(a)
    p2 = _nominal_cents(b)
    interval = p2 - p1
    abs_interval = abs(interval)

    onset_dist = min((abs(o.time_s - b.start_s) for o in onsets),
                     default=float("inf"))

    diag: dict[str, Any] = {
        "kind": "transition",
        "from_midi": a.midi,
        "to_midi": b.midi,
        "interval_cents": round(interval, 2),
        "onset_distance_ms": None if math.isinf(onset_dist) else round(onset_dist * 1000, 2),
        "transition_duration_ms": None,
        "slope_cents_per_s": None,
        "r2": None,
        "max_frame_delta_cents": None,
        "monotonic_frac": None,
        "label": "pluck",
    }

    # 1. pluck — an onset coincident with B's start wins outright
    if onset_dist <= pluck_win:
        diag["label"] = "pluck"
        return diag

    # Measure the glide. A 40-250 ms slide begins well before B's start, so the
    # analysis window looks back by the maximum slide duration (+pad), clamped
    # to A's own start, and forward by pad. _glide_span then isolates the actual
    # transition between the two stable bands regardless of where the segment
    # boundary landed.
    slide_max_s = float(art.get("slide_max_ms", 250.0)) / 1000.0
    lookback = slide_max_s + pad
    t0 = max(a.start_s, b.start_s - lookback)
    c, t = _slice(cents, times, t0, b.start_s + pad, hop_s)
    c, t = _drop_nan(c, t)
    span = _glide_span(c, p1, p2, band_tol) if len(c) >= 2 else None
    if span is None:
        diag["label"] = "pluck"   # no clean glide -> treat as a fresh pluck
        return diag

    i0, i1 = span
    gc, gt = c[i0:i1 + 1], t[i0:i1 + 1]
    # a genuine glide is continuous in time; a big gap means an unvoiced break
    # (a rest) between the notes, which is a fresh pluck, not an articulation.
    if len(gt) >= 2 and float(np.max(np.diff(gt))) > 4.0 * hop_s:
        diag["label"] = "pluck"
        return diag
    dur = float(gt[-1] - gt[0])
    diag["transition_duration_ms"] = round(dur * 1000, 2)
    max_delta = float(np.max(np.abs(np.diff(gc)))) if len(gc) >= 2 else 0.0
    diag["max_frame_delta_cents"] = round(max_delta, 2)
    slope, _, r2 = linfit_r2(gt, gc)
    diag["slope_cents_per_s"] = round(slope, 2)
    diag["r2"] = round(r2, 4)
    mono = _monotonic_frac(gc, interval)
    diag["monotonic_frac"] = round(mono, 3)

    legato_max = float(art.get("legato_max_ms", 35.0)) / 1000.0
    legato_int = float(art.get("legato_min_interval_cents", 100.0))
    legato_disc = float(art.get("legato_discontinuity_cents", 60.0))

    # 2. hammer-on / pull-off — a fast, discontinuous jump with no onset
    if dur < legato_max and abs_interval >= legato_int and max_delta > legato_disc:
        diag["label"] = "hammer_on" if interval > 0 else "pull_off"
        return diag

    slide_min = float(art.get("slide_min_ms", 40.0)) / 1000.0
    slide_max = float(art.get("slide_max_ms", 250.0)) / 1000.0
    slide_int = float(art.get("slide_min_interval_cents", 100.0))
    slide_r2 = float(art.get("slide_r2_min", 0.85))
    slide_int_max = art.get("slide_max_interval_cents", None)

    # 3. slide — a monotonic glide that establishes B's new stable band
    within_max = (slide_int_max is None) or (abs_interval <= float(slide_int_max))
    if (slide_min <= dur <= slide_max and abs_interval >= slide_int
            and within_max and r2 >= slide_r2 and mono >= 0.8):
        diag["label"] = "slide"
        return diag

    diag["label"] = "pluck"
    return diag


# --------------------------------------------------------------------------- #
# per-note interior analysis (bend / vibrato)
# --------------------------------------------------------------------------- #

def detect_bend(note: NoteEvent, cents: np.ndarray, times: np.ndarray,
                hop_s: float, cfg: Config) -> tuple[BendCurve | None, dict[str, Any]]:
    art = cfg.section("articulation")
    bend_min = float(art.get("bend_min_cents", 80.0))
    bend_max = float(art.get("bend_max_cents", 300.0))
    release_tol = float(art.get("bend_release_tol_cents", 40.0))

    c, t = _slice(cents, times, note.start_s, note.end_s, hop_s)
    c, t = _drop_nan(c, t)
    diag: dict[str, Any] = {"peak_cents": None, "released": None}
    if len(c) < 3:
        return None, diag

    base = _nominal_cents(note)
    dev = c - base
    peak_idx = int(np.argmax(dev))
    peak = float(dev[peak_idx])
    diag["peak_cents"] = round(peak, 2)
    if not (bend_min <= peak <= bend_max):
        return None, diag

    # onset of the bend: first frame the deviation exceeds half the peak
    half = peak / 2.0
    rising = np.where(dev[:peak_idx + 1] >= half)[0]
    bend_onset_s = float(t[rising[0]] - note.start_s) if rising.size else 0.0
    # hold: time spent within release_tol below the peak
    held = np.abs(dev - peak) <= release_tol
    hold_s = float(np.sum(held) * hop_s)
    # released if the contour comes back near base after the peak
    tail = dev[peak_idx:]
    released = bool(np.any(np.abs(tail) <= release_tol))
    diag["released"] = released
    return BendCurve(peak_cents=peak, onset_s=bend_onset_s, hold_s=hold_s,
                     released=released), diag


def detect_vibrato(note: NoteEvent, cents: np.ndarray, times: np.ndarray,
                   hop_s: float, cfg: Config) -> tuple[dict[str, float] | None, dict[str, Any]]:
    art = cfg.section("articulation")
    vib_min_ms = float(art.get("vibrato_min_ms", 250.0))
    f_min = float(art.get("vibrato_freq_min_hz", 4.0))
    f_max = float(art.get("vibrato_freq_max_hz", 9.0))
    d_min = float(art.get("vibrato_depth_min_cents", 20.0))
    d_max = float(art.get("vibrato_depth_max_cents", 120.0))

    diag: dict[str, Any] = {"vibrato_hz": None, "depth_cents": None}
    if note.duration_s * 1000.0 < vib_min_ms:
        return None, diag

    # Use the sustained interior (voiced frames of the note). The literal
    # "stable portion" run would fragment a wide vibrato, so we analyse the
    # full voiced contour and require a clean FFT peak (rule: reject smearing).
    c, _ = _slice(cents, times, note.start_s, note.end_s, hop_s)
    c = c[~np.isnan(c)]
    if len(c) < 8:
        return None, diag

    dev = c - float(np.mean(c))
    depth = float(np.max(c) - np.min(c))   # peak-to-peak, cents
    spec = np.abs(np.fft.rfft(dev))
    freqs = np.fft.rfftfreq(len(dev), d=hop_s)
    if len(spec) < 2:
        return None, diag
    k = int(np.argmax(spec[1:]) + 1)       # skip DC
    dom = float(freqs[k])
    # require a genuinely dominant peak, not just any modulation (Known Hard
    # Parts: separation smearing produces false vibrato)
    clean = spec[k] >= 2.0 * float(np.median(spec[1:])) if len(spec) > 2 else True
    diag["vibrato_hz"] = round(dom, 3)
    diag["depth_cents"] = round(depth, 2)
    if clean and f_min <= dom <= f_max and d_min <= depth <= d_max:
        return {"vibrato_hz": dom, "depth_cents": depth}, diag
    return None, diag


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def analyze_articulations(notes: list[NoteEvent], f0, onsets: list[Onset],
                          cfg: Config) -> tuple[list[NoteEvent], list[dict[str, Any]]]:
    """Annotate notes with entry/exit articulations, bends and vibrato.

    Returns (annotated_notes, per_note_diagnostics)."""
    cents = f0.cents
    times = np.asarray(f0.times, dtype=float)
    hop_s = float(f0.hop_s)

    out = list(notes)
    diags: list[dict[str, Any]] = [
        {"index": i, "start_s": n.start_s, "midi": n.midi,
         "entry": "pluck", "exit": None, "transition": None,
         "bend": None, "vibrato": None}
        for i, n in enumerate(notes)
    ]

    # --- transitions (sets entry of B, exit of A) ---
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        tdiag = classify_transition(a, b, cents, times, hop_s, onsets, cfg)
        diags[i]["transition"] = tdiag
        label = tdiag["label"]
        if label == "slide":
            out[i] = replace(b, entry=Articulation.SLIDE_IN)
            out[i - 1] = replace(out[i - 1], exit=Articulation.SLIDE_OUT)
            diags[i]["entry"] = "slide_in"
            diags[i - 1]["exit"] = "slide_out"
        elif label == "hammer_on":
            out[i] = replace(b, entry=Articulation.HAMMER_ON)
            diags[i]["entry"] = "hammer_on"
        elif label == "pull_off":
            out[i] = replace(b, entry=Articulation.PULL_OFF)
            diags[i]["entry"] = "pull_off"
        else:
            out[i] = replace(b, entry=Articulation.PLUCK)
            diags[i]["entry"] = "pluck"

    # --- per-note interior: bend then vibrato (respecting order) ---
    for i, note in enumerate(out):
        # A slide-out is a genuine move to a new note; never relabel it a bend.
        if note.exit != Articulation.SLIDE_OUT:
            bend, bdiag = detect_bend(note, cents, times, hop_s, cfg)
            diags[i]["bend"] = bdiag
            if bend is not None:
                exit_art = Articulation.RELEASE if bend.released else Articulation.BEND
                note = replace(note, bend=bend, exit=exit_art)
                out[i] = note
                diags[i]["exit"] = exit_art.value
                continue  # stop at first match (bend before vibrato)

        vib, vdiag = detect_vibrato(note, cents, times, hop_s, cfg)
        diags[i]["vibrato"] = vdiag
        if vib is not None:
            out[i] = replace(note, vibrato_hz=vib["vibrato_hz"],
                             vibrato_depth_cents=vib["depth_cents"])
            diags[i]["vibrato"] = {**vdiag, "labeled": True}

    return out, diags
