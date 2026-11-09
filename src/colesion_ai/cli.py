"""colesion embed | train | sweep | evaluate | export | report | data-check (see "Usage" in the README)"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from colesion_ai.core.config import load_config

REPO = Path(__file__).resolve().parents[2]


def cmd_embed(args) -> int:
    from colesion_ai.data.embeddings import cache_path, compute_embeddings

    cfg = load_config(args.config, args.overrides)
    if cache_path(cfg).exists():
        print(f"already cached: {cache_path(cfg)}")
    else:
        compute_embeddings(cfg, cache_path(cfg))
    return 0


def cmd_train(args) -> int:
    from colesion_ai.train import train  # imported here: only training needs torch

    train(load_config(args.config, args.overrides), force=args.force)
    return 0


def cmd_sweep(args) -> int:
    from colesion_ai.core.run import Run
    from colesion_ai.core.sweep import expand, run_sweep

    points = expand(args.sweep, args.overrides)
    if args.status:
        for label, cfg in points:
            run = Run(cfg)
            print(f"{run.status:<12} {run.dir.name if run.dir else '-':<26} {label}")
        return 0

    from colesion_ai.train import train

    failed = run_sweep(points, train, force=args.force)
    if failed:
        print(f"{len(failed)} failed (traceback in their log.txt):", *failed, sep="\n  ")
    return 1 if failed else 0


def cmd_evaluate(args) -> int:
    from colesion_ai.evaluate import evaluate

    evaluate(args.run, args.split)
    return 0


def cmd_export(args) -> int:
    from colesion_ai.export import export

    export(args.run)
    return 0


def cmd_report(args) -> int:
    from colesion_ai.report import report

    return 0 if report(args.paths, args.out, args.sort).height else 1


def cmd_data_check(args) -> int:
    return subprocess.call([sys.executable, str(REPO / "data" / "check_dataset.py")])


def main() -> None:
    parser = argparse.ArgumentParser(prog="colesion")
    commands = parser.add_subparsers(dest="command", required=True)
    overrides = {"nargs": "*", "help": "key.sub=value, e.g. train.lr=3e-4 run.seed=2"}
    force = {"action": "store_true", "help": "start a new run even if this config already has one"}

    embed = commands.add_parser("embed", help="cache the backbone's embeddings of all images")
    embed.add_argument("config", type=Path)
    embed.add_argument("overrides", **overrides)
    embed.set_defaults(func=cmd_embed)

    train = commands.add_parser("train", help="train one config")
    train.add_argument("config", type=Path)
    train.add_argument("overrides", **overrides)
    train.add_argument("--force", **force)
    train.set_defaults(func=cmd_train)

    sweep = commands.add_parser("sweep", help="train every grid point of a sweep")
    sweep.add_argument("sweep", type=Path)
    sweep.add_argument("overrides", **overrides)
    sweep.add_argument("--force", **force)
    sweep.add_argument("--status", action="store_true", help="only list the runs and their status")
    sweep.set_defaults(func=cmd_sweep)

    evaluate = commands.add_parser("evaluate", help="metrics + confusion matrix of a finished run")
    evaluate.add_argument("run", type=Path, help="run folder, e.g. runs/clf-frozen-dinov2/<folder>")
    evaluate.add_argument("--split", default="val", help="val (default) or, at the end, holdout")
    evaluate.set_defaults(func=cmd_evaluate)

    export = commands.add_parser("export", help="logits, embeddings and heatmaps for the dashboard")
    export.add_argument("run", type=Path, help="run folder, e.g. runs/clf-frozen-dinov2/<folder>")
    export.set_defaults(func=cmd_export)

    report = commands.add_parser("report", help="compare runs: one table (+ csv and plot)")
    report.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])
    report.add_argument("--out", type=Path, help="also write summary.csv and curves.png here")
    report.add_argument("--sort", help="column to sort by, best first, e.g. best_val_bal_acc")
    report.set_defaults(func=cmd_report)

    check = commands.add_parser("data-check", help="leak and split checks on data/build")
    check.set_defaults(func=cmd_data_check)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:  # the run has already saved its state and said so
        sys.exit(130)
