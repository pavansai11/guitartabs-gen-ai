"""Stage 2 — Separation.

Run Demucs (htdemucs) to produce four stems, cached under
``out/<song_id>/stems/``. Separation is the slowest stage — never re-run it if
the cache exists.

Target-stem selection:
  vocals  — transcribing a sung melody (the common Indian-song case)
  other   — transcribing a guitar solo
  auto    — resolved later against Stage 3 confidence (see pick_auto_source)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config

STEM_NAMES = ("vocals", "drums", "bass", "other")
# Stems worth transcribing a monophonic lead line from.
CANDIDATE_STEMS = ("vocals", "other")


def stems_dir(out_root: str | Path, song_id: str) -> Path:
    return Path(out_root) / song_id / "stems"


def cache_complete(out_root: str | Path, song_id: str) -> bool:
    d = stems_dir(out_root, song_id)
    return all((d / f"{name}.wav").exists() for name in STEM_NAMES)


def load_stem(out_root: str | Path, song_id: str, name: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    path = stems_dir(out_root, song_id) / f"{name}.wav"
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def separate(audio: np.ndarray, sr: int, song_id: str, cfg: Config,
             out_root: str | Path = "out") -> dict[str, Path]:
    """Produce (or reuse cached) Demucs stems. Returns {name: wav_path}."""
    d = stems_dir(out_root, song_id)
    paths = {name: d / f"{name}.wav" for name in STEM_NAMES}

    if cache_complete(out_root, song_id):
        return paths  # never re-run the slowest stage

    try:
        import torch
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as exc:  # pragma: no cover - requires the audio extra
        raise ImportError(
            "Stage 2 (separation) needs the audio extra: "
            "pip install 'tabforge[audio]'"
        ) from exc

    d.mkdir(parents=True, exist_ok=True)
    model = get_model(cfg.get("separation", "model", "htdemucs"))
    model.eval()

    # demucs expects (batch, channels, samples); feed stereo (duplicate mono).
    wav = torch.from_numpy(np.stack([audio, audio])).float().unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(model, wav, split=True, overlap=0.25)[0]

    src_index = {name: i for i, name in enumerate(model.sources)}
    for name in STEM_NAMES:
        stem = sources[src_index[name]].mean(dim=0).cpu().numpy()
        sf.write(str(paths[name]), stem, sr)
    return paths


def pick_auto_source(out_root: str | Path, song_id: str, cfg: Config) -> str:
    """Resolve --source auto: the candidate stem with the highest mean voiced
    confidence from Stage 3 pitch tracking."""
    from .pitch import track_pitch  # local import avoids torch at module load

    best_name, best_score = CANDIDATE_STEMS[0], -1.0
    for name in CANDIDATE_STEMS:
        audio, sr = load_stem(out_root, song_id, name)
        f0 = track_pitch(audio, sr, cfg)
        voiced = f0.confidence[f0.hz > 0]
        score = float(np.mean(voiced)) if voiced.size else 0.0
        if score > best_score:
            best_name, best_score = name, score
    return best_name
