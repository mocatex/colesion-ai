import os
import signal
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from colesion_ai.core.config import EnvCfg, RunCfg, TrainEntryCfg
from colesion_ai.core.run import Run


def toy_cfg(runs_root, seed=1) -> TrainEntryCfg:
    return TrainEntryCfg(
        run=RunCfg(name="toy", seed=seed),
        env=EnvCfg(device="cpu", runs_root=str(runs_root), data_root=str(runs_root)),
    )


def toy_training(cfg, epochs=5, stop_after_epoch=None, force=False):
    """A fake trainer that uses every kind of state a resume has to restore: weights, optimiser
    momentum, shuffling and dropout (both draw from the global RNG)."""
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    y = (x.sum(1) > 0).long()
    with Run(cfg, force=force) as run:
        model = nn.Sequential(nn.Linear(4, 16), nn.Dropout(0.3), nn.Linear(16, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        start = run.restore(model=model, optimizer=optimizer)["start_epoch"]
        for epoch in range(start, epochs):
            for batch in torch.randperm(len(x)).split(8):
                loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            run.save_checkpoint(model=model, optimizer=optimizer, epoch=epoch)
            if epoch == stop_after_epoch:
                raise KeyboardInterrupt  # like a Ctrl-C right after the checkpoint
    return start, model.state_dict()


def test_a_rerun_finds_its_folder(tmp_path):
    toy_training(toy_cfg(tmp_path), epochs=1)
    run = Run(toy_cfg(tmp_path))
    assert run.status == "done"
    assert run.dir.name.endswith("_s1")  # <start time>_s<seed>
    assert Run(toy_cfg(tmp_path, seed=2)).status == "new"  # another config: another run


def test_force_starts_a_new_run_next_to_the_old_one(tmp_path):
    toy_training(toy_cfg(tmp_path), epochs=1)
    old = Run(toy_cfg(tmp_path)).dir
    assert Run(toy_cfg(tmp_path), force=True).status == "new"

    toy_training(toy_cfg(tmp_path), epochs=1, force=True)
    new = Run(toy_cfg(tmp_path)).dir
    assert new != old and old.exists()  # same second: the new one gets a -2


def test_resume_gives_the_same_weights(tmp_path):
    _, uninterrupted = toy_training(toy_cfg(tmp_path / "a"))

    with pytest.raises(KeyboardInterrupt):
        toy_training(toy_cfg(tmp_path / "b"), stop_after_epoch=1)
    assert Run(toy_cfg(tmp_path / "b")).status == "interrupted"
    start, resumed = toy_training(toy_cfg(tmp_path / "b"))

    assert start == 2
    for name in uninterrupted:
        assert torch.equal(uninterrupted[name], resumed[name]), name


def test_sigterm_stops_the_run(tmp_path):
    with pytest.raises(KeyboardInterrupt), Run(toy_cfg(tmp_path)):
        os.kill(os.getpid(), signal.SIGTERM)  # what SLURM sends before the time limit
        time.sleep(1)  # never reached: the signal raises right away
    assert Run(toy_cfg(tmp_path)).status == "interrupted"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL  # the old handler is back


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    model = nn.Linear(2, 2)
    with Run(toy_cfg(tmp_path)) as run:
        run.save_checkpoint(model=model, epoch=0)

        def killed_mid_write(obj, path):
            Path(path).write_bytes(b"half a checkpoint")
            raise OSError("killed")

        monkeypatch.setattr(torch, "save", killed_mid_write)
        with pytest.raises(OSError):
            run.save_checkpoint(model=model, epoch=1)
        monkeypatch.undo()
        assert run.restore(model=model)["start_epoch"] == 1  # epoch 0's checkpoint is intact
