# CoLesion AI

> one of the two usecases conducted by [CAI](https://www.zhaw.ch/en/engineering/institutes-centres/cai) at ZHAW in the context of the CoLearning Project. An attempt to define a new framework for human-AI collaboration.

TODO: extend this!

## Setup

### Install dependencies

We use [uv](https://docs.astral.sh/uv/) as a package manager and script runner. See their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it for your system. Then run:

```bash
uv sync
```

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
