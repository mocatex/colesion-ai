# Data

```bash
uv run data/setup_dataset.py fetch   # download + verify everything into raw/
uv run data/setup_dataset.py build   # raw/ -> build/
```

## Layout

```text
data/
├── setup_dataset.py
├── raw/                      downloads, never edited (git-ignored)
│   ├── isic2018/             challenge images + ground truth (train 10,015 · test 1,512)
│   ├── ham10000/             lesion masks · 2020 human-AI study CSV · HAM metadata*
│   ├── barata2023/           2023 reader log
│   ├── isic_archive/         per-image metadata, train (66) + test (67)
│   └── cassidy2022/          duplicate lists*
└── build/                    THE DATASET (git-ignored, goes to Hugging Face)
    ├── BUILD.json            source URLs + checksums, seed, split counts
    ├── tables/*.parquet      4 tables, join on image_id
    ├── images/train/ISIC_0024/ISIC_0024306.jpg    10,015 (HAM10000)
    ├── images/test/ISIC_0034/ISIC_0034524.jpg      1,512 (ISIC 2018 test)
    └── masks/ISIC_0024/ISIC_0024306_segmentation.png    lesion masks, train only
```

Image and mask folders are split by the first digits of the ID (max 1,000 files each), because Hugging Face allows at most 10k files per folder. Use `image_path` / `mask_path` from `items` instead of building paths yourself.

\* fetched for reference, not read by `build`.

## Tables

| table             | one row per          |   rows | what's in it                                                         |
| :---------------- | :------------------- | -----: | :------------------------------------------------------------------- |
| `items`           | image                | 11,527 | label, metadata, file paths, human difficulty (test only), split     |
| `reader_trials`   | reader × trial       | 12,260 | Barata 2023: answer, correct, reaction time, support, unaided/aided  |
| `support_effects` | test image × support |  3,762 | Tschandl 2020: reader votes before/after each of 4 AI/crowd supports |
| `readers`         | reader               |     89 | accuracy unaided vs aided, median reaction time, accuracy over time  |

```python
import polars as pl
items = pl.read_parquet("data/build/tables/items.parquet")
```

## Splits (`items.split`)

| split     | from                         | share | used for                                                                     |
| :-------- | :--------------------------- | ----: | :--------------------------------------------------------------------------- |
| `fit`     | `ham_train` (no human data)  |  85 % | train the AI model                                                           |
| `val`     | `ham_train`                  |  15 % | model selection (checkpoints, hyperparameters)                               |
| `pool`    | `isic_test` (all human data) |  70 % | co-learning rounds: human and AI work through these, the AI learns           |
| `probe`   | `isic_test`                  |  15 % | measure the **human** alone (P_H). Never trained on                          |
| `holdout` | `isic_test`                  |  15 % | measure the **AI** alone (P_ML): the model's real test set. Never trained on |

`ham_train` has no test split on purpose: the AI is tested on `holdout`, because only `isic_test` images have human difficulty labels.

- Drawn **per lesion**: all images of one lesion land in the same split.
- Stratified on `dx` and, on test, on the `d_human` quartile, so no split is easier than another.
- Seed `2026`. Fractions and seed are constants in `setup_dataset.py`.

## Key columns

| column             | meaning                                                                                              |
| :----------------- | :--------------------------------------------------------------------------------------------------- |
| `dx`, `dx_idx`     | 7 classes, fixed order `akiec bcc bkl df mel nv vasc` = `0..6`                                       |
| `d_human`          | share of the 2020 crowd that picked the true class (1 = easy for humans). Test only                  |
| `h_count_*`        | crowd votes per class; `n_raters` is recovered from the shares (null if unanimous)                   |
| `m_prob_*`         | reference model output from the 2020 study, for sanity checks only                                   |
| `pair_id`, `phase` | each image is shown twice per reader session: `unaided` first, then `aided`                          |
| `support`          | AI shown in the aided trial: `SL` (supervised) or `RL` (reinforcement learning); `none` when unaided |

## Good to know

- Only 1,412 of the 1,512 test images have crowd votes. The other 100 have null `d_human`.
- `age`, `sex`, `localization`, `dx_type` and `lesion_id` come from the ISIC Archive for **both** splits, so they share one vocabulary.
- Images are the original JPEGs, hard-linked from `raw/`. They are not copied or re-encoded.
- Licence: CC BY-NC 4.0, non-commercial research only.
