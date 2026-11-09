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

## Classifier pipeline (phase 2)

- **Backbone -> embedding -> head.** The head is always an `nn.Linear` on embeddings. So once the backbone is fixed, everything else (head training, evaluate, export, later the loop) runs on cached embeddings, without images.
- **Embedding cache** (`data/embeddings.py`): `cache/embeddings/<backbone>_<size>_<weights>.parquet`, all 11,427 images, computed on first use.
- **Two ways to train** (`train.py`, same epoch loop):
    - `model.freeze: all`: head only, on the cache (seconds).
    - `model.freeze: none`: backbone + head on images (fine-tuning). The backbone gets `train.backbone_lr`.
- **Checkpoints** hold `head` (and `backbone` when fine-tuned). `load_run(run_dir)` returns the run's config + best head. Its config's `model.weights` then points at the backbone that makes this head's embeddings.
- **Eval transform squashes** the whole 600×450 image to a square (no crop), so heatmaps map back pixel-exactly.

## Files

| file                   | purpose                                                                                                                                                                                                                             |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cli.py`               | `colesion embed \| train \| sweep \| evaluate \| export \| report \| data-check`. Imports command modules lazily so `--help` / `--status` don't load torch.                                                                         |
| `core/config.py`       | Dataclasses (`RunCfg`…`TrainCfg`, top-level `TrainEntryCfg`). `load_config`, `dump`, `config_hash` (sha256, ignores `env`, `tracking`, `run.seed`, `run.tags`).                                                                     |
| `core/run.py`          | `Run`: one run = one folder. `restore`, `log`, `save_checkpoint`, `with run:` (status `done`/`interrupted`/`failed`/`running`), `summary`, `device`.                                                                                |
| `core/runtime.py`      | `seed_everything`, `rng_state` / `set_rng_state`, `resolve_device`.                                                                                                                                                                 |
| `core/sweep.py`        | `expand(path, overrides)` → `(label, cfg)` per grid point; `run_sweep` trains each, skips finished ones (via `train`'s own run-finding).                                                                                            |
| `train.py`             | Head on cached embeddings (`freeze: all`) or fine-tune on images (`freeze: none`); AdamW + class-weighted CE; per-epoch train/val metrics, checkpoints, `curves.png`.                                                               |
| `evaluate.py`          | `evaluate(run_dir, split)`: best head on cached embeddings → `eval/<split>/` with `metrics.json` (acc, bal_acc, macro_auc, mel_recall, ...), `per_class.csv`, `confusion.csv` + `.png`, logits; adds `<split>_*` to `summary.json`. |
| `export.py`            | `export(run_dir)`: Jonas's `export/classifier_outputs.parquet` (logits, L2-normalised embeddings, heatmap paths, all images) + `export/heatmaps/*.png` for `pool` + `holdout`.                                                      |
| `report.py`            | `collect` (table of differing config values + summary) and `plot_curves`.                                                                                                                                                           |
| `data/dataset.py`      | `load_items` (all images), `load_split`, `image_transform` (eval = squash, no crop), `ImageDataset`. `CLASSES` fixed; ImageNet normalisation.                                                                                       |
| `data/embeddings.py`   | `load_embeddings(cfg, items)` (computes + caches on first use), `cache_path`, `compute_embeddings`.                                                                                                                                 |
| `models/classifier.py` | `build_backbone` (timm; `model.weights`: pretrained / random / fine-tuned run), `build_head` (`model.head`: only `linear` so far), `load_run`, `load_checkpoint`.                                                                   |
| `models/saliency.py`   | `grad_cam(backbone, head, x)` (own ~20 lines, top-1 class) and `cam_layer` (CNN: last block; ViT: input of the last attention).                                                                                                     |
| `configs/`             | `env/laptop.yaml`; `classifier/*.yaml` (one per experiment); `sweeps/*.yaml` (`base` + `grid`).                                                                                                                                     |
| `tests/`               | 12 tests (~30 s). The important one: interrupt + resume ⇒ identical weights. Plus Grad-CAM shapes for a CNN and a ViT.                                                                                                              |

## How to use

```bash
uv run colesion train configs/classifier/smoke.yaml
uv run colesion train configs/classifier/smoke.yaml run.seed=2 --force
uv run colesion sweep configs/sweeps/smoke.yaml          # add --status to just print
uv run colesion report runs/

uv run colesion embed configs/classifier/frozen_dinov2.yaml                 # only cache embeddings
uv run colesion train configs/classifier/frozen_dinov2.yaml                 # head on the cache (seconds)
uv run colesion evaluate runs/clf-frozen-dinov2/<folder>                    # metrics + confusion matrix on val
uv run colesion export runs/clf-frozen-dinov2/<folder>                      # Jonas's parquet + heatmaps
```

- A `runs/<run.name>/<time>_s<seed>/` folder is the unit of everything.
- Re-running the same config + seed with status `done` is skipped; `--force` starts a new folder next to it.
- A sweep is just N `train` calls — no state of its own.

## How to extend

- **New setting**: add a field with a default to the relevant dataclass in `config.py`, read it as `cfg.section.field`. **IMPORTANT:** This changes `config_hash` for every config, so old runs are no longer "the same config"
- **New backbone**: one more `if` in `build_backbone` (or one more entry in `DINOV2` for a DINOv2 variant). Grad-CAM needs a `cam_layer` that fits it.
- **New metric**: return it from `classification_metrics` — it then flows into `metrics.jsonl`, `summary.json`, the plot and the report.
- **New sweep**: `configs/sweeps/<name>.yaml`:

    ```yaml
    base: ../classifier/smoke.yaml   # relative to the sweep file
    grid:
        run.seed: [1, 2, 3]
        train.lr: [1.0e-3, 3.0e-4]
    ```

    Grid keys must exist in the schema; typos fail before anything trains.
