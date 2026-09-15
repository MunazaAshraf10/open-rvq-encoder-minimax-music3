"""Linear warm up followed by linear decay to a floor (polynomial decay with power one)."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(step: int, *, warmup: int, total: int, floor: float) -> float:
    """Multiplier of the base learning rate at a given optimizer step.

    step / warmup during warm up, then (1 - floor) (1 - t) + floor with
    t = (step - warmup) / (total - warmup), and floor once the schedule ends.
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    if total <= warmup:
        return floor
    progress = min((step - warmup) / (total - warmup), 1.0)
    return (1.0 - floor) * (1.0 - progress) + floor


def make_scheduler(optimizer: Optimizer, *, warmup: int, total: int, floor: float) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: lr_lambda(step, warmup=warmup, total=total, floor=floor))
