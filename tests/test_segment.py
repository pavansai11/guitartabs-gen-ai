"""Note segmentation (Stage 5) on synthesised contours."""

from __future__ import annotations

from tabforge.config import load_config
from tabforge.models import Onset
from tabforge.stages import segment as seg
from synth import hold, rest, track


def test_two_notes_from_band_change():
    cfg = load_config()
    f0 = track(hold(0.30, 6400), hold(0.30, 6700))
    notes = seg.segment(f0, [], cfg)
    assert len(notes) == 2
    assert notes[0].midi == 64
    assert notes[1].midi == 67


def test_onset_splits_same_pitch():
    cfg = load_config()
    # one sustained pitch, but a re-pluck (onset) in the middle -> two notes
    f0 = track(hold(0.60, 6400))
    onsets = [Onset(time_s=0.30, strength=1.0)]
    notes = seg.segment(f0, onsets, cfg)
    assert len(notes) == 2
    assert all(n.midi == 64 for n in notes)


def test_short_blip_discarded():
    cfg = load_config()
    f0 = track(hold(0.30, 6400), hold(0.02, 6700), hold(0.30, 6400))
    notes = seg.segment(f0, [], cfg)
    # the 20 ms blip is below the 45 ms minimum
    assert all(n.midi == 64 for n in notes)


def test_stable_portion_sets_nominal_pitch():
    cfg = load_config()
    # long stable 6400 then a short slide tail — nominal must stay 64, not drift
    f0 = track(hold(0.25, 6400), rest(0.05))
    notes = seg.segment(f0, [], cfg)
    assert len(notes) == 1
    assert notes[0].midi == 64
