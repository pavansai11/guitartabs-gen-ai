"""MusicXML export via music21, with technical notations attached
(<slide>, <bend>, <hammer-on>, <pull-off>). Lazy import of music21."""

from __future__ import annotations

from pathlib import Path

from ..models import Articulation, Score, TabNote


def _require_music21():
    try:
        import music21  # noqa: F401
        return music21
    except ImportError as exc:  # pragma: no cover - requires notation extra
        raise ImportError(
            "MusicXML export needs music21: pip install 'tabforge[notation]'"
        ) from exc


def score_to_music21(score: Score):
    m21 = _require_music21()
    from music21 import stream, note as m21note, tempo, meter, duration, articulations, expressions

    part = stream.Part()
    part.append(tempo.MetronomeMark(number=score.tempo_bpm))
    part.append(meter.TimeSignature(f"{score.time_signature[0]}/{score.time_signature[1]}"))

    for tn in sorted(score.notes, key=lambda x: x.start_s):
        dur_ql = max(0.125, (tn.end_s - tn.start_s) * score.tempo_bpm / 60.0)
        n = m21note.Note(tn.midi)
        n.duration = duration.Duration(quarterLength=dur_ql)
        # attach technical notations
        if tn.entry == Articulation.HAMMER_ON:
            n.articulations.append(articulations.HammerOn())
        elif tn.entry == Articulation.PULL_OFF:
            n.articulations.append(articulations.PullOff())
        if tn.entry == Articulation.SLIDE_IN or tn.exit == Articulation.SLIDE_OUT:
            sl = articulations.IndeterminateSlide()
            n.articulations.append(sl)
        if tn.bend is not None:
            n.articulations.append(articulations.FretBend())
        if tn.vibrato_hz is not None:
            n.expressions.append(expressions.Tremolo())
        part.append(n)

    sc = stream.Score()
    sc.insert(0, part)
    return sc


def export_musicxml(score: Score, path: str | Path) -> Path:
    sc = score_to_music21(score)
    path = Path(path)
    sc.write("musicxml", fp=str(path))
    return path
