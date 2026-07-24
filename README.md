# TabForge — Technique-Aware Guitar Tab Generator

**Version 0.1 (MVP).** A local-first CLI that converts a single audio file into
guitar tablature that includes **expressive articulations** — slides, bends,
hammer-ons, pull-offs and vibrato — not just notes.

TabForge exists to test one hypothesis: **does contour-first articulation
detection produce tabs meaningfully better than what already exists?** Every
existing tool quantises audio to note-level MIDI early, which destroys the
continuous pitch information where articulations live. TabForge extracts a
fine-grained f0 *contour* and runs articulation analysis on the contour, before
any quantisation.

## Three rules that govern this repo

1. **The pitch contour is the product.** Articulation analysis runs on the
   continuous contour. Nothing is quantised before Stage 6.
2. **Monophonic lead lines only.** No polyphony, no chords, no multi-instrument
   scoring in v1.
3. **85% accuracy plus a fast editor beats 95% accuracy with no editor.** Every
   stage emits machine-readable intermediate artifacts so a human can inspect
   and fix them.

## ⚖️ Legal constraints — non-negotiable

- The tool processes **audio the user supplies**. It never fetches audio from
  streaming services or YouTube.
- The repository must **never** contain copyrighted audio. `data/samples/` is
  gitignored; ship only self-recorded or public-domain test clips.
- Do **not** build, and do not design toward, a hosted searchable library of
  tabs for commercial songs. Tabs of copyrighted songs are derivative works.
  The defensible structure is a *tool that transforms the user's own file*, not
  a *catalogue*.
- This constraint is intentionally in this README so it survives future
  contributors. Please keep it here.

## Why this might be better

The gap is widest for Indian film and indie vocal melodies, which are built on
*meend* — continuous glides between pitches. Western-trained models read meend
as pitch error or spurious notes. On a guitar, meend maps almost directly onto
slides. Modelling continuous pitch motion as a first-class citizen is therefore
both more accurate in general and dramatically more accurate on this repertoire.

## Install

TabForge splits its dependencies so the differentiating logic (segmentation,
articulation analysis, fretboard assignment, ASCII export, eval metrics) runs
with only `numpy`/`scipy`, while the heavy audio stack is optional.

```bash
# core logic only (no torch): enough to run tests and the fretboard/export path
pip install -e .

# full stack: separation + pitch tracking + notation + review UI
pip install -e ".[all]"
```

The heavy extras and what they unlock:

| extra       | packages                                   | unlocks                          |
| ----------- | ------------------------------------------ | -------------------------------- |
| `audio`     | soundfile, librosa, torch, torchcrepe, demucs | ingest, separation, pitch, onsets, rhythm |
| `notation`  | music21, PyGuitarPro                        | MusicXML + Guitar Pro export     |
| `review`    | fastapi, uvicorn                            | the browser review UI            |
| `viz`       | matplotlib                                  | contour PNG rendering            |
| `dev`       | pytest                                      | the test suite                   |

## Usage

```bash
# transcribe a file you own end-to-end
tabforge transcribe path/to/your_recording.mp3 --source auto --preset indian

# open the review UI for a transcribed song
tabforge review <song_id>

# re-export in a specific format
tabforge export <song_id> --format gp5

# run the evaluation harness against checked-in ground truth
tabforge eval --set data/groundtruth/
```

`transcribe` flags:

```
--source auto|vocals|other   which Demucs stem to transcribe (default auto)
--preset indian              widen slide thresholds for meend (default none)
--tuning standard            a tuning defined in config/tunings.yaml
--capo 0                     capo fret
--no-quantize                skip rhythmic quantisation entirely
--out out/                   output root
```

Every stage writes JSON to `out/<song_id>/`. `song_id` is the first 12 hex
chars of the SHA-256 of the input file's bytes, so the same file always maps to
the same output directory and caches (notably the slow separation stage) are
reused.

## Pipeline (10 stages)

| # | stage           | module                    | output                              |
|---|-----------------|---------------------------|-------------------------------------|
| 1 | Ingest          | `stages/ingest.py`        | mono/44.1k/-1dBFS audio, `song_id`  |
| 2 | Separation      | `stages/separate.py`      | Demucs stems (cached)               |
| 3 | Pitch tracking  | `stages/pitch.py`         | `F0Track` + cents contour           |
| 4 | Onset detection | `stages/onsets.py`        | `Onset[]` (pluck markers)           |
| 5 | Segmentation    | `stages/segment.py`       | `NoteEvent[]`                       |
| 6 | **Articulation**| `stages/articulation.py`  | annotated `NoteEvent[]` + diagnostics |
| 7 | Rhythm          | `stages/rhythm.py`        | tempo, beat grid, quantised starts  |
| 8 | Fretboard       | `stages/fretboard.py`     | `TabNote[]` via DP                  |
| 9 | Export          | `export/`                 | ASCII, MusicXML, `.gp5`             |
|10 | Review UI       | `review/app.py`           | local browser editor                |

The single most important discriminator: **a pitch change with a coincident
onset is a new picked note; a pitch change without one is an articulation.**
Articulation detection runs *before* fret assignment so that slide/hammer/pull
constraints (both notes on the same string) prune the fingering search toward
what a human would actually play.

## Evaluation

`tabforge eval` reports, per-excerpt and aggregate:

- **Note F1** — a match requires exact pitch and onset within 50 ms.
- **Articulation P/R/F1** per class, over correctly matched notes.
- **String/fret accuracy** — % of matched notes with the correct pair.
- **Playability** — % of transitions requiring a hand shift > 5 frets.

Ground truth lives in `data/groundtruth/*.json` (checked in — it contains no
audio, only annotations). Audio clips referenced by ground truth live in the
gitignored `data/samples/`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The core articulation logic is validated against **numpy-synthesised** signals
(known slides, bends and vibrato) in `tests/` — these run without any audio
files or torch, and are the first line of defence when tuning thresholds.

All thresholds live in `config/default.yaml` and preset overlays in
`config/presets/`. They are tunable without touching code.

## Status

This is an MVP scaffold implementing the full pipeline. The pure-logic stages
(segmentation, articulation, fretboard DP, ASCII export, eval metrics) are
complete and tested against synthesised signals. The audio-facing stages
(ingest, separation, pitch, onsets, rhythm) are implemented against the
specified libraries and activate when the `audio` extra is installed.
