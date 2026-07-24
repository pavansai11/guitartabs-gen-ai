# Ground truth

JSON annotations for the evaluation harness (`tabforge eval`). **Annotations
only — never audio.** The audio clips these refer to live in the gitignored
`data/samples/` and must be self-recorded or public-domain (see the legal
constraints in the top-level README).

## File format

Each `*.json` describes one excerpt:

```json
{
  "name": "my excerpt",
  "audio": "my_clip.wav",          // filename in data/samples/ (or null)
  "tuning": "standard",
  "tempo_bpm": 100.0,
  "source": "vocals",              // stem to transcribe (optional)
  "notes": [
    {
      "start_s": 0.0, "end_s": 0.3, "midi": 64,
      "cents_offset": 0.0, "confidence": 1.0,
      "entry": "pluck", "exit": "slide_out",
      "vibrato_hz": null, "vibrato_depth_cents": null,
      "string": 0, "fret": 0
    }
  ]
}
```

`entry`/`exit` are `Articulation` values: `pluck`, `hammer_on`, `pull_off`,
`slide_in`, `slide_out`, `bend`, `release`, `vibrato`.

## How predictions are obtained

For each ground-truth file, `tabforge eval` gets predictions by, in order:

1. loading a sibling `<name>.pred.json` if present (offline / CI path), or
2. running the pipeline on `data/samples/<audio>` when that audio is present.

`demo.json` + `demo.pred.json` are a synthetic, copyright-free example generated
by `python scripts/gen_demo.py`; they let the harness (and CI) run end-to-end
with no audio checked in.

## Making your own

Create 10 excerpts of 8–16 bars — a mix of Indian film vocal lines and Western
guitar solos. Hand-tab them (or with your tutor) and store one JSON each here,
with the corresponding clip in `data/samples/`.
