"""TabForge CLI (typer entrypoint).

    tabforge transcribe <audio> [--source auto|vocals|other] [--preset indian]
                        [--tuning standard] [--capo 0] [--no-quantize] [--out out/]
    tabforge review <song_id>
    tabforge export <song_id> --format {ascii,musicxml,gp5}
    tabforge eval [--set data/groundtruth/]
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Technique-aware guitar tab generator.")


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, readable=True,
                                 help="Audio file you own (mp3/wav/flac/m4a)."),
    source: str = typer.Option("auto", help="Target stem: auto|vocals|other."),
    preset: str = typer.Option(None, help="Config preset, e.g. 'indian'."),
    tuning: str = typer.Option("standard", help="Tuning from config/tunings.yaml."),
    capo: int = typer.Option(0, help="Capo fret."),
    style: str = typer.Option("auto", help="Fingering: auto|single|dp "
                              "(single = whole melody on one string)."),
    string: int = typer.Option(1, help="String index for --style single "
                               "(0=high e, 1=B)."),
    quantize: bool = typer.Option(True, "--quantize/--no-quantize",
                                  help="Rhythmic quantisation."),
    out: Path = typer.Option(Path("out"), help="Output root directory."),
) -> None:
    """Transcribe an audio file end-to-end into tablature."""
    from .config import load_config
    from .pipeline import transcribe as run

    cfg = load_config(preset=preset)
    strategy = {"auto": None, "single": "single_string", "dp": "dp"}.get(style)
    typer.echo(f"Transcribing {audio} (source={source}, preset={preset}, "
               f"style={style}) ...")
    result = run(audio, cfg=cfg, out_root=out, source=source, tuning_name=tuning,
                 capo=capo, quantize=quantize, preset=preset,
                 strategy=strategy, melody_string=string)
    typer.echo(f"song_id: {result.song_id}")
    typer.echo(f"notes: {len(result.score.notes)}  "
               f"(octave-folded: {len(result.folded)})")
    typer.echo(f"artifacts: {result.out_dir}")
    typer.echo("")
    with open(result.out_dir / "tab.txt", "r", encoding="utf-8") as fh:
        typer.echo(fh.read())


@app.command()
def review(
    song_id: str = typer.Argument(..., help="song_id from a previous transcribe."),
    out: Path = typer.Option(Path("out"), help="Output root directory."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Launch the local browser review UI for a transcribed song."""
    from .review.app import serve
    serve(song_id, out_root=out, host=host, port=port)


@app.command()
def export(
    song_id: str = typer.Argument(...),
    format: str = typer.Option("ascii", "--format", help="ascii|musicxml|gp5."),
    out: Path = typer.Option(Path("out")),
) -> None:
    """Re-export a stored score in a specific format."""
    from .pipeline import reexport
    path = reexport(song_id, format, out_root=out)
    typer.echo(f"wrote {path}")


@app.command()
def eval(
    set: Path = typer.Option(Path("data/groundtruth"), "--set",
                             help="Ground-truth directory."),
    samples: Path = typer.Option(Path("data/samples")),
    out: Path = typer.Option(Path("out")),
    preset: str = typer.Option(None),
) -> None:
    """Run the evaluation harness and print per-excerpt + aggregate metrics."""
    from .eval.run_eval import run_eval, format_table
    report = run_eval(gt_dir=set, samples_dir=samples, out_root=out, preset=preset)
    typer.echo(format_table(report))
    scored = [r for r in report["rows"] if r.get("status") == "ok"]
    if scored and not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
