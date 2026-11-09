import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class RunCfg:
    name: str = MISSING  # e.g. "clf-frozen-dinov2"
    seed: int = 42
    tags: list[str] = field(default_factory=list)


@dataclass
class EnvCfg:  # where it runs; not part of the run hash
    device: str = "auto"  # auto | cpu | mps | cuda
    num_workers: int = 4
    data_root: str = "data/build"
    runs_root: str = "runs"
    cache_root: str = "cache"


@dataclass
class TrackingCfg:
    log_every_n_steps: int = 50


@dataclass
class DataCfg:
    image_size: int = 224
    max_fit: int | None = None  # use a fixed random subset of `fit` (smoke runs); None = all
    max_val: int | None = None


@dataclass
class ModelCfg:
    backbone: str = "dinov2_vits14"
    # pretrained (published weights) | random (smoke runs) | a run folder that fine-tuned this backbone
    weights: str = "pretrained"
    # all: train only the head, on cached embeddings (seconds) | none: fine-tune backbone + head on images
    freeze: str = "all"
    head: str = "linear"


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3  # head
    backbone_lr: float = 1e-5  # backbone, only when fine-tuning (freeze: none)
    batch_size: int = 256
    class_weighting: str = "inverse_freq"
    weight_decay: float = 0.0


@dataclass
class TrainEntryCfg:
    """top-level config for `colesion train`"""

    run: RunCfg = field(default_factory=RunCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


# ----------------------------------------------------
# Loader
# ----------------------------------------------------


def _load_yaml(path: Path) -> DictConfig:
    """load one YAML file, recursively merging any parents listed in its `inherit` field."""
    path = Path(path).resolve()
    raw = OmegaConf.load(path)
    parents = raw.pop("inherit", [])  # inherit part is unecessary after merging
    merged = OmegaConf.create()
    for parent in parents:
        merged = OmegaConf.merge(merged, _load_yaml(path.parent / parent))
    return OmegaConf.merge(merged, raw)


def load_config(path, overrides=(), schema=TrainEntryCfg):
    """YAML (+ inherit) + CLI overrides -> validated dataclass instance
    order: schema defaults < YAML files < CLI overrides"""
    file_cfg = _load_yaml(path)
    cli_cfg = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(OmegaConf.structured(schema), file_cfg, cli_cfg)
    return OmegaConf.to_object(merged)  # catch MISSING


# ----------------------------------------------------
# Dump and hash
# ----------------------------------------------------


def dump(cfg) -> str:
    """Dataclass -> YAML text"""
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


HASH_IGNORE = ("env", "tracking", "run.seed", "run.tags")


def config_hash(cfg) -> str:
    """hash of everything that can change the result of a run."""
    d = OmegaConf.to_container(OmegaConf.structured(cfg))  # plain nested dict
    for key in HASH_IGNORE:
        *parents, last = key.split(".")
        node = d
        for p in parents:
            node = node[p]
        node.pop(last, None)
    blob = json.dumps(d, sort_keys=True)  # sort_keys: key order must not matter
    return hashlib.sha256(blob.encode()).hexdigest()
