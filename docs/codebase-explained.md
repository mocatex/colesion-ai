# Codebase guide

## Everything a new trainer needs

- **Config**: dataclasses in `core/config.py` are the schema; YAML fills them. `load_config(path, overrides)` = defaults < YAML (`inherit`) < CLI overrides, then type-checked. Unknown keys / bad types fail here.
- **Run**: `Run(cfg)` finds an earlier run with the same config hash + seed (or starts a new folder). Trainer uses exactly four things:

    ```python
    with run:                  # creates folder, writes config.yaml/meta.json, seeds, catches Ctrl-C
        run.restore(**objs)    # loads last.pt if present -> start_epoch; restores RNG states
        run.log(metrics, step=, epoch=, split=)
        run.save_checkpoint(epoch=, is_best=, **state)
    ```

    `run.summary` is a dict the trainer fills; it becomes `summary.json`.
- **A run folder** (`runs/<run.name>/<time>_s<seed>/`) holds: `config.yaml`, `meta.json`, `git.diff`, `log.txt`, `metrics.jsonl`, `summary.json`, `checkpoints/{last,best}.pt`, `artifacts/`, `curves.png`.

That's all a new trainer needs. How signals are caught, RNG is saved, and old runs are found is a black box (tested by `test_resume_gives_the_same_weights`).

## Files

| file                   | purpose                                                                                                                                                         |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cli.py`               | `colesion train \| sweep \| report \| data-check`. Imports command modules lazily so `--help` / `--status` don't load torch.                                    |
| `core/config.py`       | Dataclasses (`RunCfg`…`TrainCfg`, top-level `TrainEntryCfg`). `load_config`, `dump`, `config_hash` (sha256, ignores `env`, `tracking`, `run.seed`, `run.tags`). |
| `core/run.py`          | `Run`: one run = one folder. `restore`, `log`, `save_checkpoint`, `with run:` (status `done`/`interrupted`/`failed`/`running`), `summary`, `device`.            |
| `core/runtime.py`      | `seed_everything`, `rng_state` / `set_rng_state`, `resolve_device`.                                                                                             |
| `core/sweep.py`        | `expand(path, overrides)` → `(label, cfg)` per grid point; `run_sweep` trains each, skips finished ones (via `train`'s own run-finding).                        |
| `train.py`             | Phase-1 trainer: AdamW + CE, logs `batch_loss` every `tracking.log_every_n_steps`, per-epoch val metrics, checkpoints, `curves.png`, `predictions_val.parquet`. |
| `report.py`            | `collect` (table of differing config values + summary) and `plot_curves`.                                                                                       |
| `data/dataset.py`      | `load_split`, `image_transform`, `ImageDataset`. `CLASSES` fixed; ImageNet normalisation.                                                                       |
| `models/classifier.py` | `build_backbone` (timm, one `if` per name), `Classifier`, `build_classifier`.                                                                                   |
| `configs/`             | `env/laptop.yaml`; `classifier/*.yaml` (one per experiment); `sweeps/*.yaml` (`base` + `grid`).                                                                 |
| `tests/`               | 10 tests (~4s). The important one: interrupt + resume ⇒ identical weights.                                                                                      |

## How to use

```bash
uv run colesion train configs/classifier/smoke.yaml
uv run colesion train configs/classifier/smoke.yaml run.seed=2 --force
uv run colesion sweep configs/sweeps/smoke.yaml          # add --status to just print
uv run colesion report runs/
```

- A `runs/<run.name>/<time>_s<seed>/` folder is the unit of everything.
- Re-running the same config + seed with status `done` is skipped; `--force` starts a new folder next to it.
- A sweep is just N `train` calls — no state of its own.

## How to extend

- **New setting**: add a field with a default to the relevant dataclass in `config.py`, read it as `cfg.section.field`. **IMPORTANT:** This changes `config_hash` for every config, so old runs are no longer "the same config"
- **New backbone**: one more `if name == "...": return timm.create_model(...)` in `build_backbone`.
- **New metric**: return it from `classification_metrics` — it then flows into `metrics.jsonl`, `summary.json`, the plot and the report.
- **New sweep**: `configs/sweeps/<name>.yaml`:

    ```yaml
    base: ../classifier/smoke.yaml   # relative to the sweep file
    grid:
        run.seed: [1, 2, 3]
        train.lr: [1.0e-3, 3.0e-4]
    ```

    Grid keys must exist in the schema; typos fail before anything trains.
