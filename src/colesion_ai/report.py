"""Compare runs, from their folders alone: `colesion report` (table) and `plot_curves` (PNG)."""

import logging
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf

from colesion_ai.core.run import read_json

log = logging.getLogger(__name__)

# distinct line colours, always assigned in this order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def report(paths, out: Path | None = None) -> pl.DataFrame:
    """Print one row per run below `paths`. With `out`, also write summary.csv and curves.png."""
    table = collect(paths)
    if table.is_empty():
        log.warning("no runs found below %s", ", ".join(map(str, paths)))
        return table
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=1000,
        float_precision=4,
    ):
        print(table)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        table.write_csv(out / "summary.csv")
        logs = sorted({m for p in paths for m in Path(p).rglob("metrics.jsonl")})
        plot_curves([m.parent for m in logs if m.stat().st_size], out / "curves.png")
        log.info("wrote %s and %s", out / "summary.csv", out / "curves.png")
    return table


def collect(paths) -> pl.DataFrame:
    """One row per run: run, status, the config values that differ between the runs (e.g. what a
    sweep varies) and everything in summary.json."""
    rows, config_keys = [], set()
    for meta in sorted({m for p in paths for m in Path(p).rglob("meta.json")}):
        run_dir = meta.parent
        config = flatten(OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml")))
        config_keys |= config.keys()
        status = read_json(meta).get("status")
        summary = read_json(run_dir / "summary.json")
        rows.append({"run": label(run_dir), "status": status, **config, **summary})
    table = pl.from_dicts(rows, infer_schema_length=None)
    return table.drop(key for key in config_keys if table[key].n_unique() == 1)


def plot_curves(run_dirs: list[Path], path: Path) -> None:
    """One panel per metric logged per epoch (train loss, val acc, ...), one line per run."""
    if len(run_dirs) > len(COLORS):
        log.warning("plotting the first %d of %d runs", len(COLORS), len(run_dirs))
        run_dirs = run_dirs[: len(COLORS)]
    names = [label(d) for d in run_dirs]
    rows = pl.concat(
        pl.read_ndjson(d / "metrics.jsonl").with_columns(run=pl.lit(name))
        for d, name in zip(run_dirs, names)
    )
    rows = (
        # batch_loss is logged per step: too noisy, the per-epoch train loss is plotted instead
        rows.filter(pl.col("name") != "batch_loss")
        .with_columns(panel=pl.concat_str("split", "name", separator=" "))
        # an epoch redone after a resume appears twice: keep the redone one
        .unique(["run", "panel", "epoch"], keep="last", maintain_order=True)
    )
    panels = rows["panel"].unique(maintain_order=True)

    fig = Figure(figsize=(3.5 * len(panels), 3), layout="constrained")
    for ax, panel in zip(fig.subplots(1, len(panels), squeeze=False)[0], panels):
        for color, name in zip(COLORS, names):
            curve = rows.filter(panel=panel, run=name).sort("epoch")
            ax.plot(curve["epoch"], curve["value"], color=color, marker="o", ms=4, label=name)
        ax.set_title(panel, loc="left", fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    if len(run_dirs) > 1:
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=4)
    else:
        fig.suptitle(names[0], fontsize=8)
    fig.savefig(path, dpi=150)


def label(run_dir: Path) -> str:
    """runs/smoke/2026-10-23_15-40-47_s42 -> smoke/2026-10-23_15-40-47_s42"""
    return f"{run_dir.parent.name}/{run_dir.name}"


def flatten(d: dict, prefix: str = "") -> dict:
    """{"train": {"lr": 0.001}} -> {"train.lr": 0.001}; lists become text (for csv)."""
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[prefix + key] = str(value) if isinstance(value, list) else value
    return flat
