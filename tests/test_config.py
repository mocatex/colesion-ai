import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from colesion_ai.core.config import TrainEntryCfg, config_hash, dump, load_config


def write(path, text):
    path.write_text(text)
    return path


def test_yaml_round_trip(tmp_path):
    cfg = load_config(write(tmp_path / "a.yaml", "run: {name: x}\ntrain: {lr: 3.0e-4}"))
    again = load_config(write(tmp_path / "dumped.yaml", dump(cfg)))
    assert isinstance(again, TrainEntryCfg)
    assert again == cfg


def test_unknown_key_fails(tmp_path):
    with pytest.raises(ConfigKeyError):
        load_config(write(tmp_path / "a.yaml", "run: {name: x}\ntrain: {learning_rate: 0.1}"))


def test_wrong_type_fails(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path / "a.yaml", "run: {name: x}"), ["train.epochs=many"])


def test_inherit_order(tmp_path):
    write(tmp_path / "a.yaml", "train: {lr: 1.0, epochs: 1, batch_size: 1}")
    write(tmp_path / "b.yaml", "train: {lr: 2.0, epochs: 2}")
    child = write(
        tmp_path / "c.yaml", "inherit: [a.yaml, b.yaml]\nrun: {name: x}\ntrain: {lr: 3.0}"
    )
    cfg = load_config(child, ["train.lr=4.0"])
    # later parents beat earlier ones, the file beats its parents, the command line beats all
    assert (cfg.train.batch_size, cfg.train.epochs, cfg.train.lr) == (1, 2, 4.0)


def test_hash_ignores_key_order_and_env(tmp_path):
    a = load_config(write(tmp_path / "a.yaml", "run: {name: x}\ntrain: {lr: 0.1, epochs: 3}"))
    b = load_config(write(tmp_path / "b.yaml", "train: {epochs: 3, lr: 0.1}\nrun: {name: x}"))
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) == config_hash(load_config(tmp_path / "a.yaml", ["env.device=cpu"]))
    assert config_hash(a) != config_hash(load_config(tmp_path / "a.yaml", ["train.lr=0.2"]))
