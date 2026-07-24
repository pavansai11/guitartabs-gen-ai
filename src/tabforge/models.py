"""Data model for TabForge.

Frozen dataclasses passed between stages. Every stage serialises its output to
JSON in ``out/<song_id>/``. Serialisation helpers here convert numpy arrays to
plain lists and enums to their string values so the artifacts are portable and
human-inspectable (rule 3: build for correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


class Articulation(str, Enum):
    """How a note is entered or left."""

    PLUCK = "pluck"          # normal picked attack
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    BEND = "bend"
    RELEASE = "release"
    VIBRATO = "vibrato"


@dataclass(frozen=True)
class F0Track:
    """Fine-grained fundamental-frequency contour (Stage 3)."""

    times: np.ndarray        # seconds, uniform hop
    hz: np.ndarray           # 0.0 where unvoiced
    confidence: np.ndarray   # 0..1
    hop_s: float

    @property
    def cents(self) -> np.ndarray:
        """Continuous pitch in cents: 1200*log2(hz/440)+6900.

        One semitone == 100 cents, and cents/100 == MIDI note number.
        NaN where unvoiced (hz <= 0).
        """
        hz = np.asarray(self.hz, dtype=float)
        out = np.full(hz.shape, np.nan, dtype=float)
        voiced = hz > 0
        out[voiced] = 1200.0 * np.log2(hz[voiced] / 440.0) + 6900.0
        return out

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "times": np.asarray(self.times, dtype=float).tolist(),
            "hz": np.asarray(self.hz, dtype=float).tolist(),
            "confidence": np.asarray(self.confidence, dtype=float).tolist(),
            "hop_s": float(self.hop_s),
        }

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> "F0Track":
        return cls(
            times=np.asarray(obj["times"], dtype=float),
            hz=np.asarray(obj["hz"], dtype=float),
            confidence=np.asarray(obj["confidence"], dtype=float),
            hop_s=float(obj["hop_s"]),
        )


@dataclass(frozen=True)
class Onset:
    """A detected attack — marks a pluck (Stage 4)."""

    time_s: float
    strength: float


@dataclass(frozen=True)
class BendCurve:
    """Parametrisation of a string bend (Stage 6)."""

    peak_cents: float        # max deviation above nominal pitch
    onset_s: float           # when the bend starts, relative to note start
    hold_s: float
    released: bool = False


@dataclass(frozen=True)
class NoteEvent:
    """A single note with its articulation annotations (Stages 5-6)."""

    start_s: float
    end_s: float
    midi: int                # nominal (quantised) pitch
    cents_offset: float      # median deviation of stable portion
    confidence: float
    entry: Articulation = Articulation.PLUCK   # how the note is ENTERED
    exit: Articulation | None = None           # how it is LEFT
    bend: BendCurve | None = None
    vibrato_hz: float | None = None
    vibrato_depth_cents: float | None = None
    # raw (pre-quantisation) start, always preserved (Stage 7 never discards it)
    raw_start_s: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class TabNote(NoteEvent):
    """A NoteEvent placed on the fretboard (Stage 8)."""

    string: int = -1         # 0 = high E; -1 = unassigned
    fret: int = -1

    @classmethod
    def from_note(cls, note: NoteEvent, string: int, fret: int) -> "TabNote":
        base = asdict(note)
        # asdict recurses into nested dataclasses; rebuild them as objects.
        bend = None if note.bend is None else note.bend
        base["bend"] = bend
        base["entry"] = note.entry
        base["exit"] = note.exit
        return cls(string=string, fret=fret, **base)


@dataclass(frozen=True)
class Score:
    """The finished, playable transcription (Stage 8 output)."""

    notes: list[TabNote]
    tempo_bpm: float
    time_signature: tuple[int, int]
    tuning: list[int]        # MIDI pitch per string, high to low
    capo: int = 0


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def _encode(value: Any) -> Any:
    """Recursively encode dataclasses/enums/ndarrays into JSON-safe objects."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"__type__": type(value).__name__}
        for f in fields(value):
            out[f.name] = _encode(getattr(value, f.name))
        return out
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def note_to_dict(note: NoteEvent) -> dict[str, Any]:
    """Encode a NoteEvent or TabNote to a JSON-safe dict."""
    return _encode(note)


def dict_to_note(obj: dict[str, Any]) -> NoteEvent:
    """Reconstruct a NoteEvent/TabNote from a dict produced by note_to_dict."""
    bend = obj.get("bend")
    bend_obj = None
    if bend is not None:
        bend_obj = BendCurve(
            peak_cents=float(bend["peak_cents"]),
            onset_s=float(bend["onset_s"]),
            hold_s=float(bend["hold_s"]),
            released=bool(bend.get("released", False)),
        )

    def _art(v: Any) -> Articulation | None:
        return None if v is None else Articulation(v)

    common = dict(
        start_s=float(obj["start_s"]),
        end_s=float(obj["end_s"]),
        midi=int(obj["midi"]),
        cents_offset=float(obj["cents_offset"]),
        confidence=float(obj["confidence"]),
        entry=_art(obj.get("entry", "pluck")) or Articulation.PLUCK,
        exit=_art(obj.get("exit")),
        bend=bend_obj,
        vibrato_hz=obj.get("vibrato_hz"),
        vibrato_depth_cents=obj.get("vibrato_depth_cents"),
        raw_start_s=obj.get("raw_start_s"),
    )
    if "string" in obj and "fret" in obj and obj["string"] is not None:
        return TabNote(string=int(obj["string"]), fret=int(obj["fret"]), **common)
    return NoteEvent(**common)


def score_to_dict(score: Score) -> dict[str, Any]:
    return {
        "__type__": "Score",
        "notes": [note_to_dict(n) for n in score.notes],
        "tempo_bpm": float(score.tempo_bpm),
        "time_signature": list(score.time_signature),
        "tuning": list(score.tuning),
        "capo": int(score.capo),
    }


def dict_to_score(obj: dict[str, Any]) -> Score:
    notes = [dict_to_note(n) for n in obj["notes"]]
    tab_notes = [n if isinstance(n, TabNote) else TabNote.from_note(n, -1, -1)
                 for n in notes]
    ts = obj["time_signature"]
    return Score(
        notes=tab_notes,
        tempo_bpm=float(obj["tempo_bpm"]),
        time_signature=(int(ts[0]), int(ts[1])),
        tuning=[int(x) for x in obj["tuning"]],
        capo=int(obj.get("capo", 0)),
    )
