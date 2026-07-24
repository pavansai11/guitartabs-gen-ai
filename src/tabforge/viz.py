"""Contour visualisation.

The pitch contour plotted with detected note boundaries is THE debugging tool
that makes everything else tractable (see Stage 10). Rendered as a static PNG
for milestone checks and served interactively by the review UI.

matplotlib is imported lazily (viz extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import F0Track, NoteEvent


def render_contour_png(f0: F0Track, notes: list[NoteEvent], path: str | Path,
                       title: str = "TabForge pitch contour") -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - requires viz extra
        raise ImportError(
            "Contour PNG needs matplotlib: pip install 'tabforge[viz]'"
        ) from exc

    cents = f0.cents
    times = np.asarray(f0.times, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, cents, lw=0.8, color="#1f77b4", label="f0 (cents)")

    for n in notes:
        ax.axvline(n.start_s, color="#d62728", lw=0.5, alpha=0.6)
        nominal = n.midi * 100.0 + n.cents_offset
        ax.hlines(nominal, n.start_s, n.end_s, color="#2ca02c", lw=1.5)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (cents; 100 = 1 semitone)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path
