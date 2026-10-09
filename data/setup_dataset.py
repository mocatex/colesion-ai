#!/usr/bin/env python3
"""Fetch every source and build the co-learning dataset.

uv run data/setup_dataset.py fetch   # download + verify into data/raw/ write data/build/BUILD.json
uv run data/setup_dataset.py build   # data/raw/ -> data/build/ (tables/*.parquet, images/, masks/)
uv run data/setup_dataset.py all
"""

import argparse
import json
import math
import random
import shutil
import subprocess
from datetime import UTC, datetime
from itertools import accumulate
from pathlib import Path
from urllib.parse import unquote

import polars as pl
import polars.selectors as cs
import pooch
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BUILD = ROOT / "build"

# FIXED dx_idx order, written into the dataset card.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

ISIC_2018 = "https://isic-challenge-data.s3.amazonaws.com/2018/"
CASSIDY = "https://raw.githubusercontent.com/mmu-dermatology-research/isic_duplicate_removal_strategy/main/file_lists/"
DATAVERSE = "https://dataverse.harvard.edu"
ISIC_API = "https://api.isic-archive.com/api/v2"

# Plain downloads: folder -> {url: sha256}. None skips the check
# copy a file sha256 from BUILD.json into slot to pin it
HTTP_SOURCES = {
    "isic2018": {
        ISIC_2018 + "ISIC2018_Task3_Training_Input.zip": None,
        ISIC_2018 + "ISIC2018_Task3_Training_GroundTruth.zip": None,
        ISIC_2018 + "ISIC2018_Task3_Training_LesionGroupings.csv": None,
        ISIC_2018 + "ISIC2018_Task3_Test_Input.zip": None,
        ISIC_2018 + "ISIC2018_Task3_Test_GroundTruth.zip": None,
    },
    # Cassidy et al. 2022 duplicate lists.
    "cassidy2022": {
        CASSIDY + "06%20-%20all_train_duplicates_deleted_(all%20but%20newest).txt": None,
        CASSIDY + "12%20-%20all_test_duplicates_deleted_(all%20but%20newest).txt": None,
    },
}

# Dataverse records: folder -> (DOI, original filenames)
DATAVERSE_SOURCES = {
    "ham10000": (
        "doi:10.7910/DVN/DBW86T",
        [
            "HAM10000_metadata",
            "HAM10000_segmentations_lesion_tschandl.zip",
            "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv",
        ],
    ),
    "barata2023": ("doi:10.7910/DVN/PWQMQ7", ["reader_data_nature_medicine.csv"]),
}

# ISIC Archive collections for test images (not in HAM10000)
ISIC_COLLECTIONS = {"train": 66, "test": 67}

# Barata 2023 reader log: shipped column -> schema name.
READER_COLUMNS = {
    "test_id": "session_id",
    "tequ_user_id": "reader_id",
    "image_isic_id": "image_id",
    "tequ_no": "trial_no",
    "tequ_time": "rt_sec",
    "tequ_correct": "correct",
    "answer": "answer_dx",
    "probType": "support",
    "image_diag_id": "true_dx",
}


# --- fetch -----------------------------------------------------------------


def get_json(url: str, **params) -> dict:
    r = requests.get(url, params=params, headers={"User-Agent": "colesion-ai"}, timeout=60)
    r.raise_for_status()
    return r.json()


def describe(path: Path, url: str) -> dict:
    """One BUILD.json entry"""
    return {
        str(path.relative_to(ROOT)): {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": pooch.file_hash(path),
        }
    }


def download(folder: str, url: str, known_hash: str | None = None, fname: str | None = None) -> dict:
    """Cached checksum-verified download. Zips are also extracted to <name>.zip.unzip/"""
    fname = fname or unquote(url.rsplit("/", 1)[-1])
    unzip = pooch.Unzip() if fname.endswith(".zip") else None
    pooch.retrieve(url, known_hash, fname=fname, path=RAW / folder, processor=unzip, progressbar=True)
    return describe(RAW / folder / fname, url)


def dataverse_files(doi: str) -> dict[str, dict]:
    """Original filename -> file record, for the latest version of a Dataverse dataset."""
    version = get_json(f"{DATAVERSE}/api/datasets/:persistentId/", persistentId=doi)["data"]["latestVersion"]
    files = [f["dataFile"] for f in version["files"]]
    return {f.get("originalFileName", f["filename"]): f for f in files}


def isic_metadata(split: str, collection: int) -> dict:
    """Write one ISIC Archive collection's image metadata to CSV (paginated API, 100 per page)."""
    dest = RAW / "isic_archive" / f"collection_{collection}_{split}.csv"
    url = f"{ISIC_API}/images/search/?collections={collection}&limit=100"
    if not dest.exists():
        rows, page = [], url
        while page:
            data = get_json(page)
            for img in data["results"]:
                row = {
                    "isic_id": img["isic_id"],
                    "license": img["copyright_license"],
                    "attribution": img["attribution"],
                }
                for group in img["metadata"].values():  # acquisition, clinical
                    row |= group
                rows.append(row)
            page = data["next"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows, infer_schema_length=None).write_csv(dest)
    return describe(dest, url)


def write_manifest(**fields) -> None:
    """Merge fields into BUILD.json and stamp it with the time, git commit and script hash."""
    path = BUILD / "BUILD.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest |= {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git.stdout.strip() or None,
        "script_sha256": pooch.file_hash(__file__),
        "classes": CLASSES,
        **fields,
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {path}")


def fetch() -> None:
    sources = {}
    for folder, files in HTTP_SOURCES.items():
        for url, sha256 in files.items():
            sources |= download(folder, url, sha256)

    for folder, (doi, names) in DATAVERSE_SOURCES.items():
        files = dataverse_files(doi)
        for name in names:
            if name not in files:
                raise KeyError(f"{name!r} not in {doi}. Available: {sorted(files)}")
            url = f"{DATAVERSE}/api/access/datafile/{files[name]['id']}?format=original"
            sources |= download(folder, url, f"md5:{files[name]['md5']}", fname=name)

    for split, collection in ISIC_COLLECTIONS.items():
        sources |= isic_metadata(split, collection)

    write_manifest(sources=sources)


# ---------------------------------------------------------------------------
# --- build -----------------------------------------------------------------
# ---------------------------------------------------------------------------

SEED = 2026

# Split fractions per image source. Splits are drawn per lesion, so a lesion never straddles two splits.
SPLITS = {
    "ham_train": {"fit": 0.85, "val": 0.15},
    "isic_test": {"pool": 0.8, "holdout": 0.2},
}

CROWD_SHARES = [f"prob_h_dx_{c}" for c in CLASSES]

# ISIC Archive diagnosis_confirm_type -> HAM10000 dx_type (matches 1:1 on the training images)
DX_TYPES = {
    "histopathology": "histo",
    "serial imaging showing no change": "follow_up",
    "single image expert consensus": "consensus",
    "confocal microscopy with consensus dermoscopy": "confocal",
}

# 2020 CSV interaction_modality -> short name
MODALITIES = {
    "Malignancy Probability (AI)": "malignancy_prob",
    "Multiclass Probabilities (AI)": "multiclass_ai",
    "CBIR (AI)": "cbir",
    "Multiclass Probabilities (Crowd)": "multiclass_crowd",
}


def unzipped(folder: str, name: str) -> Path:
    """Where pooch extracted <name>.zip"""
    return RAW / folder / f"{name}.zip.unzip" / name


IMAGE_DIRS = {
    "ham_train": unzipped("isic2018", "ISIC2018_Task3_Training_Input"),
    "isic_test": unzipped("isic2018", "ISIC2018_Task3_Test_Input"),
}
MASK_DIR = unzipped("ham10000", "HAM10000_segmentations_lesion_tschandl")


def at_dx(prefix: str) -> pl.Expr:
    """Per row, the column <prefix><dx>: the value for the true class."""
    return pl.coalesce([pl.when(pl.col("dx") == c).then(pl.col(prefix + c)) for c in CLASSES])


def read_labels() -> pl.DataFrame:
    """Challenge ground truth (one-hot CSVs) -> image_id, split_source, dx."""
    labels = []
    for source, name in [
        ("ham_train", "ISIC2018_Task3_Training_GroundTruth"),
        ("isic_test", "ISIC2018_Task3_Test_GroundTruth"),
    ]:
        onehot = pl.read_csv(unzipped("isic2018", name) / f"{name}.csv")
        labels.append(
            onehot.unpivot(index="image", variable_name="dx")
            .filter(pl.col("value") == 1)
            .select(image_id="image", split_source=pl.lit(source), dx=pl.col("dx").str.to_lowercase())
        )
    return pl.concat(labels)


def read_archive_metadata() -> pl.DataFrame:
    """lesion_id / dx_type / age / sex / site for train AND test, so both splits share one vocabulary.
    (Same lesion grouping as HAM10000_metadata, but cleaner: no placeholder ages, fewer unknown sexes.)"""
    files = sorted((RAW / "isic_archive").glob("collection_*.csv"))
    meta = pl.concat([pl.read_csv(f, infer_schema_length=None) for f in files], how="diagonal_relaxed")
    return meta.select(
        image_id="isic_id",
        lesion_id="lesion_id",
        dx_type=pl.col("diagnosis_confirm_type").replace_strict(DX_TYPES),
        age=pl.col("age_approx").cast(pl.Int8),
        sex="sex",
        localization="anatom_site_1",
    )


def read_crowd_csv(path: Path) -> pl.DataFrame:
    """2020 reader CSV: comma decimals inside quoted fields. Crowd shares sum to ~1 if parsed right."""
    df = pl.read_csv(path, decimal_comma=True)
    total = df.select(pl.sum_horizontal(CROWD_SHARES)).to_series()
    if not total.is_between(0.9, 1.0 + 1e-6).all():
        raise ValueError(f"crowd shares sum to {total.min():.3f}..{total.max():.3f}; the parse is wrong")
    return df


def recover_n(shares, max_n: int = 400, tol: float = 4e-3) -> int | None:
    """Smallest N that turns every crowd share into a near-integer vote count. None if unanimous (any N fits)."""
    shares = [s for s in shares if s is not None]
    if all(s in (0, 1) for s in shares):
        return None
    for n in range(2, max_n):
        if all(abs(s * n - round(s * n)) < tol for s in shares):
            return n
    return None


def crowd_per_image(crowd: pl.DataFrame) -> pl.DataFrame:
    """Per test image: crowd votes and the reference model's probabilities (Tschandl 2020)."""
    per_image = crowd.unique("image_id", keep="first")  # shares + model probs repeat on all 4 modality rows
    n_raters = [recover_n(row) for row in per_image.select(CROWD_SHARES).rows()]
    plogp = [pl.when(pl.col(s) > 0).then(pl.col(s) * pl.col(s).log()).otherwise(0.0) for s in CROWD_SHARES]
    return per_image.with_columns(n_raters=pl.Series(n_raters, dtype=pl.Int16)).select(
        "image_id",
        "n_raters",
        *CROWD_SHARES,  # kept until d_human is computed
        *[
            (pl.col(f"prob_h_dx_{c}") * pl.col("n_raters")).round().cast(pl.Int16).alias(f"h_count_{c}")
            for c in CLASSES
        ],
        *[pl.col(f"prob_m_dx_{c}").cast(pl.Float32).alias(f"m_prob_{c}") for c in CLASSES],
        m_prob_mal=pl.col("prob_m_mal").cast(pl.Float32),
        h_top1=pl.concat_list(CROWD_SHARES).list.arg_max().replace_strict(dict(enumerate(CLASSES))),
        h_entropy=(-pl.sum_horizontal(plogp) / math.log(len(CLASSES))).cast(pl.Float32),  # 0 => unanimous, 1 => uniform
    )


def assign_splits(items: pl.DataFrame) -> pl.DataFrame:
    """Seeded split per lesion, stratified on dx and (test only) the d_human quartile,
    so no split ends up systematically easier than another."""
    lesions = (
        items.group_by("split_source", "lesion_id")
        .agg(pl.col("dx").first(), pl.col("d_human").mean())
        .sort("lesion_id")
        .with_columns(d_quartile=pl.col("d_human").qcut(4, allow_duplicates=True).cast(pl.String))
    )
    strata = ["split_source", "dx", "d_quartile"]  # d_quartile is null on train
    # one reproducible random number per lesion, so adding/removing lesions never moves the others
    draw = [random.Random(f"{SEED}:{lesion_id}").random() for lesion_id in lesions["lesion_id"]]
    pos = (pl.col("draw").rank().over(strata) - 0.5) / pl.len().over(strata)  # random position in [0, 1) per stratum

    def split_by_position(fractions: dict[str, float]) -> pl.Expr:
        return pos.cut(list(accumulate(fractions.values()))[:-1], labels=list(fractions)).cast(pl.String)

    lesions = lesions.with_columns(draw=pl.Series(draw)).with_columns(
        split=pl.coalesce([pl.when(pl.col("split_source") == s).then(split_by_position(f)) for s, f in SPLITS.items()])
    )
    return items.join(lesions.select("split_source", "lesion_id", "split"), on=["split_source", "lesion_id"])


def build_items(crowd: pl.DataFrame) -> pl.DataFrame:
    """One row per image."""
    folder = {"ham_train": "train", "isic_test": "test"}
    shard = pl.col("image_id").str.head(9)  # ISIC_0024306 -> ISIC_0024: max 1k files per folder (HF allows 10k)
    items = (
        read_labels()
        .join(read_archive_metadata(), on="image_id", how="left")
        .join(crowd_per_image(crowd), on="image_id", how="left")
        .with_columns(
            dx_idx=pl.col("dx").replace_strict(CLASSES, list(range(len(CLASSES))), return_dtype=pl.Int8),
            d_human=at_dx("prob_h_dx_").cast(pl.Float32),  # = h_count[dx] / n_raters, also defined if unanimous
            image_path=pl.format(
                "images/{}/{}/{}.jpg", pl.col("split_source").replace_strict(folder), shard, "image_id"
            ),
            mask_path=pl.when(pl.col("split_source") == "ham_train").then(
                pl.format("masks/{}/{}_segmentation.png", shard, "image_id")
            ),
        )
        # approx. 100 images without crowd votes are dropped
        .filter((pl.col("split_source") == "ham_train") | pl.col("d_human").is_not_null())
    )
    return (
        assign_splits(items)
        .select(
            "image_id",
            "split_source",
            "lesion_id",
            "dx",
            "dx_idx",
            "dx_type",
            "age",
            "sex",
            "localization",
            "image_path",
            "mask_path",
            "n_raters",
            cs.starts_with("h_count_"),
            "d_human",
            "h_top1",
            "h_entropy",
            cs.starts_with("m_prob_"),
            "split",
        )
        .sort("image_id")
    )


def read_reader_log(path: Path) -> pl.DataFrame:
    """One row per reader-trial (Barata 2023). Each image is shown twice per session: unaided, then aided.
    Label the pair; report, never drop, odd groups."""
    key = ["session_id", "image_id"]
    rank = pl.col("trial_no").rank("ordinal").over(key)
    trials = (
        pl.read_csv(path)
        .rename(READER_COLUMNS)
        .with_columns(
            pl.col("answer_dx", "true_dx").str.to_lowercase(),
            trial_no=pl.col("trial_no").cast(pl.Int8),
            rt_sec=pl.col("rt_sec").cast(pl.Float32),
            correct=pl.col("correct").cast(pl.Boolean),
            support=pl.col("support").replace("no help", "none"),
            pair_id=pl.concat_str(key, separator=":"),
            phase=pl.when(rank == 1).then(pl.lit("unaided")).when(rank == 2).then(pl.lit("aided")),
        )
    )
    odd = trials.group_by(key).len().filter(pl.col("len") != 2)
    if odd.height:
        print(f"WARNING: {odd.height} session/image groups are not pairs (sizes {odd['len'].unique().to_list()})")
    return trials


def build_readers(trials: pl.DataFrame) -> pl.DataFrame:
    """One row per reader: what the mocked reader is fitted against."""
    thirds = [(1, 7), (8, 14), (15, 20)]
    return (
        trials.group_by("reader_id")
        .agg(
            pl.col("profession", "age_group", "gender").first(),
            n_trials=pl.len(),
            acc_unaided=pl.col("correct").filter(pl.col("phase") == "unaided").mean(),
            acc_aided=pl.col("correct").filter(pl.col("phase") == "aided").mean(),
            median_rt_sec=pl.col("rt_sec").median(),
            acc_by_trial_third=pl.concat_list(
                [pl.col("correct").filter(pl.col("trial_no").is_between(a, b)).mean() for a, b in thirds]
            ),
        )
        .sort("reader_id")
    )


def build_support_effects(crowd: pl.DataFrame, items: pl.DataFrame) -> pl.DataFrame:
    """One row per test image x support modality (Tschandl 2020): reader votes before and after the support."""
    before = {f"user_dx_without_interaction_{c}": f"before_{c}" for c in CLASSES}
    after = {f"user_dx_with_interaction_{c}": f"after_{c}" for c in CLASSES}
    return (
        crowd.rename(before | after)
        .join(items.select("image_id", "dx"), on="image_id")
        .with_columns(
            modality=pl.col("interaction_modality").replace_strict(MODALITIES),
            n_before=pl.sum_horizontal(list(before.values())),
            n_after=pl.sum_horizontal(list(after.values())),
            correct_before=at_dx("before_"),
            correct_after=at_dx("after_"),
        )
        .select(
            "image_id",
            "modality",
            pl.col("n_before", "n_after").cast(pl.Int16),
            cs.starts_with("before_", "after_").cast(pl.Int16),
            pl.col("correct_before", "correct_after").cast(pl.Int16),
        )
        .sort("image_id", "modality")
    )


def link_images(items: pl.DataFrame) -> None:
    """Hard-link images and masks into build/: instant, no extra disk, bytes exactly as shipped."""
    for old in ("images", "masks"):  # start clean so no stale files from an older layout get uploaded
        shutil.rmtree(BUILD / old, ignore_errors=True)
    for image_id, source, image_path, mask_path in items.select(
        "image_id", "split_source", "image_path", "mask_path"
    ).iter_rows():
        pairs = [(IMAGE_DIRS[source] / f"{image_id}.jpg", BUILD / image_path)]
        if mask_path:
            pairs.append((MASK_DIR / f"{image_id}_segmentation.png", BUILD / mask_path))
        for src, dst in pairs:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.hardlink_to(src)


def check_difficulty(items: pl.DataFrame) -> float:
    """Step 4: d_human should be spread out, and only weakly tied to the model's own difficulty."""
    voted = items.filter(pl.col("d_human").is_not_null())
    print(f"d_human over {voted.height} test images with crowd votes:")
    for _, label, count in voted["d_human"].hist(bins=[i / 10 for i in range(11)]).iter_rows():
        print(f"  {label:>10} {count:5d} {'#' * (count // 10)}")
    rho = voted.select(pl.corr("d_human", at_dx("m_prob_"), method="spearman")).item()
    print(f"Spearman rho(d_human, model prob of true class) = {rho:.3f}")
    return rho


def build() -> None:
    crowd = read_crowd_csv(RAW / "ham10000/ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv")
    trials = read_reader_log(RAW / "barata2023/reader_data_nature_medicine.csv")
    items = build_items(crowd)
    tables = {
        "items": items,
        "reader_trials": trials,
        "support_effects": build_support_effects(crowd, items),
        "readers": build_readers(trials),
    }

    (BUILD / "tables").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(BUILD / "tables" / f"{name}.parquet")
        print(f"wrote tables/{name}.parquet ({df.height} rows)")
    link_images(items)

    write_manifest(
        build={
            "seed": SEED,
            "split_fractions": SPLITS,
            "split_counts": dict(items.group_by("split").len().sort("split").iter_rows()),
            "rows": {name: df.height for name, df in tables.items()},
            "spearman_d_human_vs_model": check_difficulty(items),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fetch", "build", "all"])
    stage = parser.parse_args().stage
    if stage in ("fetch", "all"):
        fetch()
    if stage in ("build", "all"):
        build()


if __name__ == "__main__":
    main()
