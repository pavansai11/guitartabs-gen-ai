"""Guitar Pro .gp5 export via PyGuitarPro, with beat effects populated.

This is the format that matters most to real guitarists — it plays back, so the
user can hear whether the transcription is right. Rhythm mapping here is
approximate (one beat per note, duration snapped to the nearest power-of-two
subdivision); the review UI is where fine rhythmic correction happens.

PyGuitarPro is imported lazily (notation extra).
"""

from __future__ import annotations

from pathlib import Path

from ..models import Articulation, Score, TabNote

# Allowed note-duration values in Guitar Pro (whole=1 .. thirty-second=32).
_DURATION_VALUES = [1, 2, 4, 8, 16, 32]


def _require_guitarpro():
    try:
        import guitarpro
        return guitarpro
    except ImportError as exc:  # pragma: no cover - requires notation extra
        raise ImportError(
            "Guitar Pro export needs PyGuitarPro: pip install 'tabforge[notation]'"
        ) from exc


def _duration_value(dur_s: float, tempo_bpm: float) -> int:
    """Snap a duration in seconds to the nearest GP duration value."""
    beats = max(1e-3, dur_s * tempo_bpm / 60.0)   # in quarter-note beats
    ideal = 4.0 / beats                            # value where quarter==4
    return min(_DURATION_VALUES, key=lambda v: abs(v - ideal))


def build_song(score: Score):
    gp = _require_guitarpro()

    song = gp.Song()
    song.tempo = int(round(score.tempo_bpm))

    track = song.tracks[0]
    track.name = "TabForge"
    # strings: GP numbers strings 1..N from high to low; tuning is high->low.
    track.strings = [gp.GuitarString(number=i + 1, value=v)
                     for i, v in enumerate(score.tuning)]

    beats_per_measure = score.time_signature[0]
    ordered = sorted(score.notes, key=lambda x: x.start_s)

    # Group notes into measures by accumulated quarter-note length.
    measures: list[list[TabNote]] = [[]]
    acc = 0.0
    for tn in ordered:
        val = _duration_value(tn.end_s - tn.start_s, score.tempo_bpm)
        length_in_beats = 4.0 / val
        if acc + length_in_beats > beats_per_measure and measures[-1]:
            measures.append([])
            acc = 0.0
        measures[-1].append(tn)
        acc += length_in_beats

    # Ensure the song has enough measure headers/measures.
    while len(song.measureHeaders) < len(measures):
        song.addMeasureHeader(gp.MeasureHeader())
    # rebuild track measures to match
    track.measures = []
    for header, group in zip(song.measureHeaders, measures):
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        voice.beats = []
        for tn in group:
            beat = gp.Beat(voice)
            beat.duration = gp.Duration(
                value=_duration_value(tn.end_s - tn.start_s, score.tempo_bpm))
            note = gp.Note(beat)
            note.value = int(tn.fret)
            note.string = int(tn.string) + 1        # GP strings are 1-indexed
            note.effect = _note_effect(gp, tn)
            beat.notes = [note]
            voice.beats.append(beat)
        track.measures.append(measure)
    return song


def _note_effect(gp, tn: TabNote):
    effect = gp.NoteEffect()
    if tn.entry == Articulation.HAMMER_ON or tn.entry == Articulation.PULL_OFF:
        effect.hammer = True
    if tn.entry == Articulation.SLIDE_IN or tn.exit == Articulation.SLIDE_OUT:
        try:
            effect.slides = [gp.SlideType.shiftSlideTo]
        except Exception:  # pragma: no cover - API variance across versions
            pass
    if tn.vibrato_hz is not None:
        effect.vibrato = True
    if tn.bend is not None:
        bend = gp.BendEffect()
        # a simple two-point bend up to the detected peak (in quarter-tone units)
        peak_quarter = max(1, int(round(tn.bend.peak_cents / 50.0)))
        bend.type = gp.BendType.bend
        bend.value = peak_quarter * gp.GPFileBase.bendPosition if hasattr(
            gp, "GPFileBase") else peak_quarter
        try:
            bend.points = [
                gp.BendPoint(position=0, value=0),
                gp.BendPoint(position=6, value=peak_quarter),
                gp.BendPoint(position=12, value=0 if tn.bend.released else peak_quarter),
            ]
        except Exception:  # pragma: no cover
            pass
        effect.bend = bend
    return effect


def export_gp5(score: Score, path: str | Path) -> Path:
    gp = _require_guitarpro()
    song = build_song(score)
    path = Path(path)
    gp.write(song, str(path))
    return path
