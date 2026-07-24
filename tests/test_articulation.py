"""Core articulation tests on synthesised signals (Milestone 3).

Each test builds a known slide / bend / vibrato / hammer / pull with numpy and
asserts the classifier labels it correctly — the first line of defence before
any real audio, and the discriminator between the failure modes the spec calls
out (notably slide-misread-as-bend)."""

from __future__ import annotations

import numpy as np
import pytest

from tabforge.config import load_config
from tabforge.models import Articulation, Onset
from tabforge.stages import segment as seg
from tabforge.stages.articulation import (analyze_articulations,
                                          classify_transition, detect_bend,
                                          detect_vibrato)
from synth import hold, note, ramp, rest, track, vibrato, HOP_S


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def cfg_indian():
    return load_config(preset="indian")


# --------------------------------------------------------------------------- #
# transitions: pluck / slide / hammer / pull
# --------------------------------------------------------------------------- #

def test_slide_up_classified(cfg):
    f0 = track(hold(0.20, 6400), ramp(0.12, 6400, 6700), hold(0.20, 6700))
    a = note(0.0, 0.32, 64)
    b = note(0.32, 0.52, 67)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] == "slide"
    assert 40 <= diag["transition_duration_ms"] <= 250
    assert diag["r2"] >= 0.85


def test_slide_down_classified(cfg):
    f0 = track(hold(0.20, 6700), ramp(0.12, 6700, 6400), hold(0.20, 6400))
    a = note(0.0, 0.32, 67)
    b = note(0.32, 0.52, 64)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] == "slide"


def test_pluck_when_onset_present(cfg):
    # same geometry as a slide, but an onset at B's start makes it a fresh pluck
    f0 = track(hold(0.20, 6400), ramp(0.12, 6400, 6700), hold(0.20, 6700))
    a = note(0.0, 0.32, 64)
    b = note(0.32, 0.52, 67)
    onsets = [Onset(time_s=0.32, strength=1.0)]
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, onsets, cfg)
    assert diag["label"] == "pluck"


def test_hammer_on_classified(cfg):
    f0 = track(hold(0.20, 6400), hold(0.20, 6700))   # instant jump, no ramp
    a = note(0.0, 0.20, 64)
    b = note(0.20, 0.40, 67)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] == "hammer_on"
    assert diag["max_frame_delta_cents"] > 60


def test_pull_off_classified(cfg):
    f0 = track(hold(0.20, 6700), hold(0.20, 6400))
    a = note(0.0, 0.20, 67)
    b = note(0.20, 0.40, 64)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] == "pull_off"


def test_rest_between_notes_is_pluck_not_slide(cfg):
    # an unvoiced gap (a rest) between two pitches must not read as a glide
    f0 = track(hold(0.20, 6400), rest(0.10), hold(0.20, 6700))
    a = note(0.0, 0.20, 64)
    b = note(0.30, 0.50, 67)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] == "pluck"


# --------------------------------------------------------------------------- #
# bend / release
# --------------------------------------------------------------------------- #

def test_bend_and_release_detected(cfg):
    f0 = track(hold(0.15, 6400), ramp(0.05, 6400, 6600),
               ramp(0.05, 6600, 6400), hold(0.15, 6400))
    n = note(0.0, 0.40, 64)
    bend, diag = detect_bend(n, f0.cents, f0.times, HOP_S, cfg)
    assert bend is not None
    assert 80 <= bend.peak_cents <= 300
    assert bend.released is True


def test_flat_note_has_no_bend(cfg):
    f0 = track(hold(0.30, 6400))
    n = note(0.0, 0.30, 64)
    bend, _ = detect_bend(n, f0.cents, f0.times, HOP_S, cfg)
    assert bend is None


# --------------------------------------------------------------------------- #
# vibrato
# --------------------------------------------------------------------------- #

def test_vibrato_detected(cfg):
    f0 = track(vibrato(0.40, 6400, freq_hz=5.0, depth_cents=60.0))
    n = note(0.0, 0.40, 64)
    vib, diag = detect_vibrato(n, f0.cents, f0.times, HOP_S, cfg)
    assert vib is not None
    assert 4.0 <= vib["vibrato_hz"] <= 9.0
    assert 20.0 <= vib["depth_cents"] <= 120.0


def test_short_note_no_vibrato(cfg):
    f0 = track(vibrato(0.15, 6400, freq_hz=5.0, depth_cents=60.0))
    n = note(0.0, 0.15, 64)
    vib, _ = detect_vibrato(n, f0.cents, f0.times, HOP_S, cfg)
    assert vib is None   # below the 250 ms minimum duration


# --------------------------------------------------------------------------- #
# the hard case: slide vs bend
# --------------------------------------------------------------------------- #

def test_slide_not_relabeled_as_bend_end_to_end(cfg):
    """A slide establishes a new stable band at its destination; it must be a
    slide, and the source note's excursion must NOT be reported as a bend."""
    f0 = track(hold(0.20, 6400), ramp(0.12, 6400, 6700), hold(0.20, 6700))
    onsets = []
    notes = seg.segment(f0, onsets, cfg)
    assert len(notes) == 2
    out, _ = analyze_articulations(notes, f0, onsets, cfg)
    assert out[1].entry == Articulation.SLIDE_IN
    assert out[0].exit == Articulation.SLIDE_OUT
    assert out[0].bend is None            # never a bend
    assert out[1].bend is None


def test_indian_preset_widens_slide(cfg_indian):
    # 350 ms meend glide across 5 semitones — too long/wide for the default
    # slide window, accepted under the Indian preset.
    f0 = track(hold(0.20, 6000), ramp(0.35, 6000, 6500), hold(0.20, 6500))
    a = note(0.0, 0.55, 60)
    b = note(0.55, 0.75, 65)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg_indian)
    assert diag["label"] == "slide"


def test_indian_wide_slide_rejected_by_default(cfg):
    f0 = track(hold(0.20, 6000), ramp(0.35, 6000, 6500), hold(0.20, 6500))
    a = note(0.0, 0.55, 60)
    b = note(0.55, 0.75, 65)
    diag = classify_transition(a, b, f0.cents, f0.times, HOP_S, [], cfg)
    assert diag["label"] != "slide"      # 350 ms exceeds default 250 ms ceiling
