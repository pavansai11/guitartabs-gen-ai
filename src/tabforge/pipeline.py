"""Pipeline orchestration.

Runs the stages in order and writes a machine-readable artifact for each into
``out/<song_id>/`` (rule 3: every stage emits inspectable, correctable output).
Heavy audio stages are imported lazily so importing this module does not require
torch/librosa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, get_tuning, load_config
from .export.ascii_tab import render_ascii_tab
from .models import (F0Track, NoteEvent, Onset, Score, note_to_dict,
                     score_to_dict, dict_to_score)
from .stages import articulation as art_stage
from .stages import fretboard as fret_stage
from .stages import segment as seg_stage


@dataclass
class TranscribeResult:
    song_id: str
    f0: F0Track
    onsets: list[Onset]
    notes: list[NoteEvent]          # articulated, pre-fretboard
    score: Score
    diagnostics: list[dict[str, Any]]
    folded: list[int]
    out_dir: Path


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def transcribe(audio_path: str | Path, cfg: Config | None = None,
               out_root: str | Path = "out", source: str = "auto",
               tuning_name: str = "standard", capo: int = 0,
               quantize: bool = True, preset: str | None = None,
               write_artifacts: bool = True) -> TranscribeResult:
    """Full transcription: audio file -> Score, writing artifacts per stage."""
    # audio-facing stages imported here so the pure path stays torch-free
    from .stages.ingest import ingest
    from .stages.separate import (separate, load_stem, pick_auto_source,
                                  CANDIDATE_STEMS)
    from .stages.pitch import track_pitch
    from .stages.onsets import detect_onsets
    from .stages.rhythm import apply_rhythm

    cfg = cfg or load_config(preset=preset)
    out_root = Path(out_root)

    # Stage 1 — ingest
    ing = ingest(audio_path, cfg, out_root=out_root)
    song_id = ing.song_id
    out_dir = out_root / song_id
    if write_artifacts:
        _write_json(out_dir / "ingest.json", {
            "song_id": song_id, "source_path": str(ing.source_path),
            "sr": ing.sr, "n_samples": int(ing.audio.shape[0]),
        })

    # Stage 2 — separation (cached)
    separate(ing.audio, ing.sr, song_id, cfg, out_root=out_root)

    # resolve target stem
    if source == "auto":
        source = pick_auto_source(out_root, song_id, cfg)
    stem_audio, stem_sr = load_stem(out_root, song_id, source)

    # Stage 3 — pitch
    f0 = track_pitch(stem_audio, stem_sr, cfg)
    if write_artifacts:
        _write_json(out_dir / "pitch.json", f0.to_json_obj())

    # Stage 4 — onsets
    onsets = detect_onsets(stem_audio, stem_sr, cfg)
    if write_artifacts:
        _write_json(out_dir / "onsets.json",
                    [{"time_s": o.time_s, "strength": o.strength} for o in onsets])

    # Stage 5 — segmentation
    notes = seg_stage.segment(f0, onsets, cfg)
    if write_artifacts:
        _write_json(out_dir / "segments.json", [note_to_dict(n) for n in notes])

    # Stage 6 — articulation (⭐)
    notes, diagnostics = art_stage.analyze_articulations(notes, f0, onsets, cfg)
    if write_artifacts:
        _write_json(out_dir / "articulation.json", {
            "notes": [note_to_dict(n) for n in notes],
            "diagnostics": diagnostics,
        })

    # Stage 7 — rhythm
    notes, grid = apply_rhythm(notes, stem_audio, stem_sr, cfg, quantize=quantize)

    # Stage 8 — fretboard
    tuning = get_tuning(tuning_name)
    score, folded = fret_stage.build_score(
        notes, tuning, cfg, tempo_bpm=grid.tempo_bpm, capo=capo)

    result = TranscribeResult(
        song_id=song_id, f0=f0, onsets=onsets, notes=notes, score=score,
        diagnostics=diagnostics, folded=folded, out_dir=out_dir)

    if write_artifacts:
        _write_json(out_dir / "score.json", score_to_dict(score))
        _write_json(out_dir / "meta.json", {
            "song_id": song_id, "source": source, "tuning": tuning_name,
            "capo": capo, "quantize": quantize, "preset": preset,
            "tempo_bpm": grid.tempo_bpm, "beat_confidence": grid.confidence,
            "octave_folded_note_indices": folded,
        })
        with open(out_dir / "tab.txt", "w", encoding="utf-8") as fh:
            fh.write(render_ascii_tab(score, cfg))
        try:
            from .viz import render_contour_png
            render_contour_png(f0, notes, out_dir / "contour.png",
                               title=f"TabForge — {song_id}")
        except ImportError:
            pass
    return result


def load_score(song_id: str, out_root: str | Path = "out") -> Score:
    path = Path(out_root) / song_id / "score.json"
    with open(path, "r", encoding="utf-8") as fh:
        return dict_to_score(json.load(fh))


def reexport(song_id: str, fmt: str, out_root: str | Path = "out",
             cfg: Config | None = None) -> Path:
    """Re-export a stored score to the requested format."""
    cfg = cfg or load_config()
    score = load_score(song_id, out_root=out_root)
    out_dir = Path(out_root) / song_id
    if fmt == "ascii":
        path = out_dir / "tab.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_ascii_tab(score, cfg))
        return path
    if fmt == "musicxml":
        from .export.musicxml import export_musicxml
        return export_musicxml(score, out_dir / "score.musicxml")
    if fmt == "gp5":
        from .export.gp5 import export_gp5
        return export_gp5(score, out_dir / "score.gp5")
    raise ValueError(f"Unknown format '{fmt}' (ascii|musicxml|gp5)")
