import torch

from colesion_ai.train import lr_schedule


def test_cosine_warms_up_then_falls_to_zero():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = lr_schedule(optimizer, "cosine", steps_per_epoch=10, epochs=5)
    lrs = []
    for _ in range(50):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0.1 and lrs[9] == 1.0  # warm-up over the first epoch
    assert lrs[9:] == sorted(lrs[9:], reverse=True)  # then only down
    assert lrs[-1] < 0.01
