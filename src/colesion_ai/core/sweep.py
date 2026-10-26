"""A sweep is a base config plus a grid. Every grid point is an ordinary `colesion train` run,
so rerunning a sweep skips the finished runs and resumes the interrupted one.

    base: ../classifier/smoke.yaml   # relative to the sweep file, like `inherit`
    grid:
      run.seed: [1, 2]
      train.lr: [1.0e-3, 3.0e-4]     # 2 x 2 = 4 runs
"""

import itertools
import logging
from pathlib import Path

from omegaconf import OmegaConf

from colesion_ai.core.config import load_config
from colesion_ai.core.run import Run

log = logging.getLogger(__name__)


def expand(path, overrides=()) -> list[tuple[str, object]]:
    """(label, config) per grid point, e.g. ("run.seed=1 train.lr=0.001", cfg).
    The config is exactly what `colesion train <base> <label> <overrides>` would get."""
    spec = OmegaConf.load(path)
    base = Path(path).parent / spec.base
    points = []
    for values in itertools.product(*spec.grid.values()):
        point = [f"{key}={value}" for key, value in zip(spec.grid.keys(), values)]
        points.append((" ".join(point), load_config(base, [*point, *overrides])))
    return points


def run_sweep(points: list, train, force: bool = False) -> list[str]:
    """`train(cfg, force=force)` for every grid point. Returns the labels of the failed ones."""
    done = sum(Run(cfg, force=force).status == "done" for _, cfg in points)
    log.info("sweep: %d runs, %d done, %d to run", len(points), done, len(points) - done)
    failed = []
    for label, cfg in points:
        try:
            train(cfg, force=force)
        # one broken grid point must not stop the others; Ctrl-C (not an Exception) still does
        except Exception as e:  # noqa: BLE001
            log.error("%s failed, going on with the next run: %s", label, e)
            failed.append(label)
    return failed
