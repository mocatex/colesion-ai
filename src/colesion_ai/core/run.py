"""One run = one folder, named by its start time: runs/<run.name>/2026-10-23_15-40-47_s42/.

Running the same config again finds that folder through the config hash in meta.json: a done run
is skipped, any other one resumes from its last finished epoch. Used by a trainer like this:

    run = Run(cfg)
    if run.status != "done":
        with run:
            state = run.restore(model=model, optimizer=optimizer)
            for epoch in range(state["start_epoch"], epochs):
                ...  # train one epoch
                run.log({"bal_acc": 0.71}, step=step, epoch=epoch, split="val")
                run.save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, step=step)
"""

import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from colesion_ai.core.config import config_hash, dump
from colesion_ai.core.runtime import resolve_device, rng_state, seed_everything, set_rng_state

log = logging.getLogger(__name__)

PACKAGES = ("torch", "torchvision", "timm", "numpy", "polars", "scikit-learn", "omegaconf")


class Run:
    def __init__(self, cfg, force: bool = False):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        # the latest earlier run of this config, or None: then __enter__ makes a new folder.
        # `force` ignores earlier runs, so a new one starts next to them.
        self.dir = None if force else find_run(cfg, self.hash)
        self.meta = read_json(self.dir / "meta.json") if self.dir else {}
        # final metrics: the trainer fills it, `colesion report` reads it
        self.summary = read_json(self.dir / "summary.json") if self.dir else {}

    @property
    def status(self) -> str:
        """new | running | interrupted | failed | done. "running" on disk means the process died
        without cleaning up (kill -9, power loss); like interrupted and failed, it resumes."""
        return self.meta.get("status", "new")

    # --- start and end ---------------------------------------------------------

    def __enter__(self):
        if self.status == "done":
            raise RuntimeError(f"{self.dir} is already done (check `run.status`, or use --force)")
        resuming = self.dir is not None
        if not resuming:
            self.dir = new_run_dir(self.cfg)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        self._log_file = attach_log_file(self.dir / "log.txt")
        log.info("%s %s", "resuming" if resuming else "starting", self.dir)

        commit = git("rev-parse", "HEAD")
        if resuming and commit != self.meta.get("git_commit"):
            log.warning("the code changed since this run started (new commit)")
        self.device = resolve_device(self.cfg.env.device)
        (self.dir / "config.yaml").write_text(dump(self.cfg))
        # uncommitted changes (tracked files only): commit + diff = the exact code of this run
        if diff := git("diff", "HEAD"):
            (self.dir / "git.diff").write_text(diff)
        now = utc_now()
        self.meta = {
            "status": "running",
            "config_hash": self.hash,
            "seed": self.cfg.run.seed,
            "started": self.meta.get("started", now),
            "resumed": self.meta.get("resumed", []) + ([now] if resuming else []),
            "ended": None,
            "git_commit": commit,
            "git_dirty": bool(git("status", "--porcelain")),
            "dataset_sha256": file_sha256(Path(self.cfg.env.data_root) / "BUILD.json"),
            "device": self.device,
            "host": platform.node(),
            "python": platform.python_version(),
            "packages": {name: package_version(name) for name in PACKAGES},
        }
        write_json(self.dir / "meta.json", self.meta)

        seed_everything(self.cfg.run.seed)
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._old_handlers = {
            sig: signal.signal(sig, self._stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        return self

    def _stop(self, signum, frame):
        """Ctrl-C and SIGTERM (what SLURM sends before it kills a job) stop the run at once.
        That is safe: resuming starts from the last checkpoint anyway, and checkpoints are
        written atomically. Repeats are ignored (`uv run` forwards SIGTERM, so it can arrive
        twice), so that __exit__ can finish."""
        for sig in self._old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            status = "done"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
            log.warning("interrupted: run the same command again to resume")
        else:
            status = "failed"
            log.error("failed: %r (traceback in log.txt)", exc)
            self._log_file.stream.write("".join(traceback.format_exception(exc)))

        self._metrics.close()
        self.meta.update(status=status, ended=utc_now())
        write_json(self.dir / "meta.json", self.meta)
        write_json(self.dir / "summary.json", self.summary)
        log.info("%s: %s", self.dir, status)
        logging.getLogger("colesion_ai").removeHandler(self._log_file)
        self._log_file.close()
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        return False  # let the exception through: Ctrl-C must also stop a sweep

    # --- metrics and checkpoints -------------------------------------------------

    def log(self, metrics: dict, *, step: int, epoch: int | None = None, split: str | None = None):
        """Append to metrics.jsonl (one row per value; plots and reports read it) and print it."""
        t = round(time.time(), 3)
        for name, value in metrics.items():
            row = {
                "step": step,
                "epoch": epoch,
                "split": split,
                "name": name,
                "value": float(value),
                "t": t,
            }
            self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()  # a killed run keeps everything logged so far
        values = "  ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        log.info("epoch %s  step %d  %-5s  %s", epoch, step, split or "", values)

    def save_checkpoint(self, *, epoch: int, is_best: bool = False, **state) -> None:
        """Write checkpoints/last.pt (and best.pt) at the end of an epoch.

        `state` holds objects with a state_dict() (model, optimiser, ...) and plain values (step,
        best score). The RNG states are added, so that a resumed run draws the same random
        numbers (shuffling, augmentation, dropout) as one that was never stopped."""
        import torch

        checkpoint = {
            "epoch": epoch,
            "rng": rng_state(),
            "state": {
                k: v.state_dict() if hasattr(v, "state_dict") else v for k, v in state.items()
            },
        }
        for name in ["last.pt", "best.pt"] if is_best else ["last.pt"]:
            path = self.dir / "checkpoints" / name
            # write + rename is atomic: a kill mid-write keeps the old file
            torch.save(checkpoint, path.with_suffix(".tmp"))
            os.replace(path.with_suffix(".tmp"), path)

    def restore(self, **objects) -> dict:
        """Load last.pt into `objects` and restore the RNG states.

        Returns the saved plain values plus `start_epoch`; just {"start_epoch": 0} if there is no
        checkpoint. An epoch cut short is redone in full."""
        path = self.dir / "checkpoints" / "last.pt"
        if not path.exists():
            return {"start_epoch": 0}
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # our own file
        state = checkpoint["state"]
        for name, obj in objects.items():
            obj.load_state_dict(state.pop(name))
        set_rng_state(checkpoint["rng"])
        log.info("restored the end of epoch %d", checkpoint["epoch"])
        return {**state, "start_epoch": checkpoint["epoch"] + 1}


# --- helpers ---------------------------------------------------------------------


def find_run(cfg, config_hash: str) -> Path | None:
    """The latest folder in runs/<run.name>/ with this config hash and seed, or None."""
    found = None
    for meta_path in sorted(Path(cfg.env.runs_root, cfg.run.name).glob("*/meta.json")):
        meta = read_json(meta_path)
        if meta.get("config_hash") == config_hash and meta.get("seed") == cfg.run.seed:
            found = meta_path.parent  # sorted by start time, so the last match is the latest
    return found


def new_run_dir(cfg) -> Path:
    """runs/<run.name>/<local start time>_s<seed>, plus -2, -3, ... if that name is taken."""
    parent = Path(cfg.env.runs_root, cfg.run.name)
    name = f"{datetime.now().astimezone():%Y-%m-%d_%H-%M-%S}_s{cfg.run.seed}"
    path, n = parent / name, 2
    while path.exists():
        path, n = parent / f"{name}-{n}", n + 1
    return path


def git(*args: str) -> str | None:
    """Output of a git command in this repo; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def attach_log_file(path: Path) -> logging.FileHandler:
    """Everything logged under `colesion_ai.*` also goes to the run's log.txt."""
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    package_logger = logging.getLogger("colesion_ai")
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(handler)
    return handler


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    path.with_suffix(".tmp").write_text(json.dumps(data, indent=2) + "\n")
    os.replace(path.with_suffix(".tmp"), path)  # atomic, like the checkpoints


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
