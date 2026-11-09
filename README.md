# CoLesion AI

> one of the two usecases conducted by [CAI](https://www.zhaw.ch/en/engineering/institutes-centres/cai) at ZHAW in the context of the CoLearning Project. An attempt to define a new framework for human-AI collaboration.

TODO: extend this!

## Setup

### Install dependencies

We use [uv](https://docs.astral.sh/uv/) as a package manager and script runner. See their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it for your system. Then run:

```bash
uv sync
```

### Setup Pre-Commit (via prek)

[prek](https://github.com/j178/prek) is a Rust based, drop-in replacement for pre-commit. Install it via your preferred way (e.g. `brew`or `mise`).
Then setup the pre-commit hooks and test them on all our current files:

```bash
prek install # installs hooks: ruff-check, ruff-format
prek run --all-files
```

If any file got changes it should "fail" the check. You can then inspect the changes and run it again. (or commit directly)
After this, the pre-commit hooks will run automatically on every commit.

### Download and build the dataset

We have two possible ways to download and build the dataset:

1. **Run the script** (recommended):

    ```bash
    uv run data/setup_dataset.py fetch   # download + verify everything into raw/
    uv run data/setup_dataset.py build   # raw/ -> build/
    ```

2. **Download the prebuilt dataset from Hugging Face**

    ```bash
    hf download mocatex/colesion-ai-dataset --repo-type dataset
    ```

## Usage

Everything goes through one command, `uv run colesion ...`. How the code works inside: [docs/codebase-explained.md](docs/codebase-explained.md).

| command                                              | what it does                                                                                                                |
| :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| `colesion embed <config.yaml> [key=value ...]`       | Caches the backbone's embeddings of all images (`train` also does this when needed).                                        |
| `colesion train <config.yaml> [key=value ...]`       | Trains one config: one run, one folder in `runs/`.                                                                          |
| `colesion sweep <sweep.yaml> [key=value ...]`        | Trains every grid point of a sweep, one after the other (= many `train` calls).                                             |
| `colesion sweep <sweep.yaml> --status`               | Lists the grid points and whether each is done. Trains nothing.                                                             |
| `colesion evaluate <run folder> [--split val]`       | Metrics, per-class table and confusion matrix of a finished run (no retraining).                                            |
| `colesion export <run folder>`                       | Logits, embeddings and Grad-CAM heatmaps for Jonas's dashboard.                                                             |
| `colesion report [folders] [--out DIR] [--sort COL]` | Compares runs: one table row per run (`--sort best_val_bal_acc` puts the best first); `--out` also writes a csv and a plot. |
| `colesion data-check`                                | Leak and split checks on `data/build/` (after every dataset build).                                                         |
| `uv run pytest`                                      | Tests, about 4 s.                                                                                                           |

### Typical workflow

```bash
# 1. After changing code: tests + the smoke run (CPU, 200 images, ~25 s, no download)
uv run pytest
uv run colesion train configs/classifier/smoke.yaml

# 2. One experiment: a YAML file in configs/, train it, look at it class by class
uv run colesion train configs/classifier/frozen_dinov2.yaml
uv run colesion train configs/classifier/frozen_dinov2.yaml train.lr=3e-4   # try one value, no new file
uv run colesion evaluate runs/clf-frozen-dinov2/<folder>                    # on val

# 3. Compare settings or seeds: a sweep file, check the plan, run it, compare
uv run colesion sweep configs/sweeps/smoke.yaml --status
uv run colesion sweep configs/sweeps/smoke.yaml
uv run colesion report runs/smoke --sort best_val_bal_acc --out runs/_reports/smoke
```

### The classifier pipeline

The backbone (image -> embedding) is the only expensive part, so images are touched only to fine-tune it, to cache its embeddings and for the heatmaps. Everything else runs on the cached embeddings in seconds.

```text
images ── train, model.freeze: none ──► fine-tuned backbone + head      optional, once (variant b)
images ── embed ──────────────────────► cache/embeddings/<backbone>_<size>_<weights>.parquet
                                          all 11,427 images, once per backbone version (~2.5 min on MPS)
cache ─── train, model.freeze: all ───► trained head (+ curves)          seconds
cache + head ── evaluate ─────────────► <run>/eval/<split>/: metrics, per-class table, confusion matrix
cache + head ── export ───────────────► <run>/export/: Jonas's parquet; images ─► Grad-CAM heatmaps
```

- `model.freeze: all` trains only the head, on the cached embeddings. `model.freeze: none` fine-tunes backbone and head together on images (`configs/classifier/finetune_dinov2.yaml`).
- `model.weights` says which backbone: `pretrained` (published weights), `random` (smoke runs) or a run folder that fine-tuned it. Example: freeze a fine-tuned backbone and train new heads on it with `model.weights=runs/clf-finetune-dinov2/<folder> model.freeze=all`.
- `evaluate` and `export` always use the run's best epoch (`checkpoints/best.pt`). Choose models on `val`; use `--split holdout` only for the final candidates.
- `export` writes, for all 11,427 images, the 7 uncalibrated logits (`logit_akiec` ... `logit_vasc`) and the L2-normalised embedding, plus a 600×450 Grad-CAM heatmap (top-1 class, turbo colours) for the `pool` and `holdout` images. Join it with `items.parquet` on `image_id`. The heatmaps belong to that run's head.

**`train` or `sweep`?** Both train. Use `train` for one config. Use `sweep` when you want the same config with a few values changed (learning rates, backbones, seeds, ...). A sweep file lists the values (`base:` config + `grid:`), and `sweep` calls `train` once per combination.

### What a run leaves behind

Every run gets a folder named by its start time and seed: `runs/<run.name>/2026-10-23_15-40-47_s42/`.

| file                                | what it is                                                                                                 |
| :---------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| `curves.png`                        | loss, acc and bal_acc per epoch, each with a train and a val curve. Redrawn after every epoch.             |
| `summary.json`                      | Final numbers (what `colesion report` shows).                                                              |
| `metrics.jsonl`                     | Every logged value, one per line.                                                                          |
| `log.txt`                           | Everything printed, plus the traceback if the run failed.                                                  |
| `config.yaml`                       | The fully resolved config; `colesion train <this file>` reproduces the run.                                |
| `meta.json`                         | Status, start/end/resume times, config hash, git commit, dataset hash, package versions, device.           |
| `git.diff`                          | Uncommitted code changes at the start (only if there were any). Commit + diff = the exact code of the run. |
| `checkpoints/last.pt`, `best.pt`    | End of the last finished epoch (resume point) / epoch with the best val balanced accuracy.                 |
| `artifacts/predictions_val.parquet` | Per-image probabilities on `val`, for any later metric.                                                    |

`colesion report` puts several runs side by side. Its table only shows the config values that differ between the runs (for a sweep: the swept ones). With `--out DIR`, it also writes `summary.csv` and one `curves.png`: colour = run, dashed = train, solid = val.

### Running the same config again

`train` and `sweep` look for an earlier run of the same config with the same seed, via the config hash in `meta.json`:

- **Done:** it is skipped. Rerunning a sweep only trains what's missing.
- **Interrupted, failed or killed:** it resumes in its own folder, from the last finished epoch.
- **`--force`:** a new run starts in a new folder; the old one stays.

"The same config" means the same values, except `env.*` (device, workers, paths), `tracking.*` and `run.tags`. These don't change the result, so `env.device=cpu` doesn't start a new run.

### Stop and resume

Ctrl-C stops a run at once and marks it `interrupted`. Running it again will resume the run from where it left off. (Same with a sweep)A run that fails with an error doesn't stop the sweep; the error is in its `log.txt`.
