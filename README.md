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

| command                                        | what it does                                                                       |
| :--------------------------------------------- | :--------------------------------------------------------------------------------- |
| `colesion train <config.yaml> [key=value ...]` | Trains one config: one run, one folder in `runs/`.                                 |
| `colesion sweep <sweep.yaml> [key=value ...]`  | Trains every grid point of a sweep, one after the other (= many `train` calls).    |
| `colesion sweep <sweep.yaml> --status`         | Lists the grid points and whether each is done. Trains nothing.                    |
| `colesion report [folders] [--out DIR]`        | Compares runs: prints one table row per run; `--out` also writes a csv and a plot. |
| `colesion data-check`                          | Leak and split checks on `data/build/` (after every dataset build).                |
| `uv run pytest`                                | Tests, about 4 s.                                                                  |

### Typical workflow

```bash
# 1. After changing code: tests + the smoke run (CPU, 200 images, ~25 s, no download)
uv run pytest
uv run colesion train configs/classifier/smoke.yaml

# 2. One experiment: a YAML file in configs/, then train it
uv run colesion train configs/classifier/frozen_dinov2.yaml
uv run colesion train configs/classifier/frozen_dinov2.yaml train.lr=3e-4   # try one value, no new file

# 3. Compare settings or seeds: a sweep file, check the plan, run it, compare
uv run colesion sweep configs/sweeps/smoke.yaml --status
uv run colesion sweep configs/sweeps/smoke.yaml
uv run colesion report runs/smoke --out runs/_reports/smoke
```

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
