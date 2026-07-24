"""Stage 10 — Review UI backend (FastAPI).

    tabforge review <song_id>

Serves the artifacts under out/<song_id>/ to a local single-page app:
  - pitch contour + detected note boundaries (the key debugging visualisation)
  - audio playback
  - click-to-edit pitch / string / fret / articulation
  - corrections saved to out/<song_id>/corrections.json (append-only; never
    overwritten — they double as future training data) and re-exported

FastAPI/uvicorn are imported lazily (review extra).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..export.ascii_tab import render_ascii_tab
from ..config import load_config
from ..models import Score, TabNote, dict_to_score, note_to_dict, score_to_dict

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _song_dir(out_root: str | Path, song_id: str) -> Path:
    return Path(out_root) / song_id


def load_corrections(song_dir: Path) -> list[dict[str, Any]]:
    path = song_dir / "corrections.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("corrections", []) if isinstance(data, dict) else data


def append_correction(song_dir: Path, correction: dict[str, Any]) -> None:
    """Append a correction. Existing corrections are never overwritten."""
    existing = load_corrections(song_dir)
    correction = {**correction, "ts": datetime.now(timezone.utc).isoformat()}
    existing.append(correction)
    with open(song_dir / "corrections.json", "w", encoding="utf-8") as fh:
        json.dump({"corrections": existing}, fh, indent=2)


def apply_corrections(score: Score, corrections: list[dict[str, Any]]) -> Score:
    from ..models import Articulation
    from dataclasses import replace
    notes = list(score.notes)
    for c in corrections:
        i = int(c.get("note_index", -1))
        if not (0 <= i < len(notes)):
            continue
        n = notes[i]
        kw: dict[str, Any] = {}
        if "midi" in c and c["midi"] is not None:
            kw["midi"] = int(c["midi"])
        if "string" in c and c["string"] is not None:
            kw["string"] = int(c["string"])
        if "fret" in c and c["fret"] is not None:
            kw["fret"] = int(c["fret"])
        if c.get("entry"):
            kw["entry"] = Articulation(c["entry"])
        if "exit" in c:
            kw["exit"] = Articulation(c["exit"]) if c["exit"] else None
        notes[i] = replace(n, **kw)
    return replace(score, notes=notes)


def corrected_score(song_dir: Path) -> Score:
    with open(song_dir / "score.json", "r", encoding="utf-8") as fh:
        base = dict_to_score(json.load(fh))
    return apply_corrections(base, load_corrections(song_dir))


def _downsample(seq: list[float], max_points: int = 6000) -> tuple[list[float], int]:
    n = len(seq)
    if n <= max_points:
        return seq, 1
    step = n // max_points + 1
    return seq[::step], step


def create_app(song_id: str, out_root: str | Path = "out"):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - requires review extra
        raise ImportError(
            "Review UI needs fastapi+uvicorn: pip install 'tabforge[review]'"
        ) from exc

    out_root = Path(out_root)
    song_dir = _song_dir(out_root, song_id)
    cfg = load_config()

    app = FastAPI(title=f"TabForge review — {song_id}")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/song")
    def song_data() -> Any:
        with open(song_dir / "pitch.json", "r", encoding="utf-8") as fh:
            pitch = json.load(fh)
        meta = {}
        meta_path = song_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        onsets = []
        op = song_dir / "onsets.json"
        if op.exists():
            onsets = json.loads(op.read_text(encoding="utf-8"))
        score = corrected_score(song_dir)
        hz, step = _downsample(pitch["hz"])
        times, _ = _downsample(pitch["times"])
        return JSONResponse({
            "song_id": song_id,
            "meta": meta,
            "hop_s": pitch["hop_s"],
            "times": times,
            "hz": hz,
            "downsample_step": step,
            "onsets": onsets,
            "notes": [note_to_dict(n) for n in score.notes],
            "tuning": score.tuning,
            "tempo_bpm": score.tempo_bpm,
            "tab": render_ascii_tab(score, cfg),
        })

    @app.get("/api/audio")
    def audio() -> Any:
        for name in ("ingest.wav", "stems/vocals.wav", "stems/other.wav"):
            p = song_dir / name
            if p.exists():
                return FileResponse(str(p), media_type="audio/wav")
        return JSONResponse({"error": "no audio artifact"}, status_code=404)

    @app.post("/api/correct")
    async def correct(request: Request) -> Any:
        payload = await request.json()
        append_correction(song_dir, payload)
        score = corrected_score(song_dir)
        # re-export ASCII on every correction
        (song_dir / "tab.txt").write_text(render_ascii_tab(score, cfg), encoding="utf-8")
        return JSONResponse({
            "ok": True,
            "notes": [note_to_dict(n) for n in score.notes],
            "tab": render_ascii_tab(score, cfg),
        })

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def serve(song_id: str, out_root: str | Path = "out",
          host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Review UI needs uvicorn: pip install 'tabforge[review]'"
        ) from exc
    app = create_app(song_id, out_root=out_root)
    print(f"TabForge review UI:  http://{host}:{port}   (song_id={song_id})")
    uvicorn.run(app, host=host, port=port)
