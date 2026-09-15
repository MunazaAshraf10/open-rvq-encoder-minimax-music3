import torch
import torch.nn.functional as F
from torch import Tensor

from conftest import TINY_VOCABS, tiny_config
from rvq_ae.constants import IGNORE
from rvq_ae.evaluate import Counter, evaluate
from rvq_ae.model import RvqEncoder


class Controlled(RvqEncoder):
    """Semantic head always right; acoustic heads right only under teacher forcing."""

    def __init__(self) -> None:
        super().__init__(tiny_config())
        self.answers = torch.zeros(0)

    def logits(self, features: Tensor, targets: Tensor | None = None) -> list[Tensor]:
        codes = self.answers
        out = [F.one_hot(codes[:, :, 0].clamp(min=0), TINY_VOCABS[0]).float() * 10]
        for book in range(1, 8):
            column = codes[:, :, book].clamp(min=0) if targets is not None else (codes[:, :, book] + 1) % 5
            out.append(F.one_hot(column, TINY_VOCABS[book]).float() * 10)
        return out


def batches(model: Controlled, count: int) -> list[dict[str, Tensor]]:
    return [
        {
            "latents": torch.randn(2, 12, 128),
            "pool": torch.rand(2, 4, 12),
            "target": torch.stack([torch.randint(0, vocab, (2, 4)) for vocab in TINY_VOCABS], dim=-1),
        }
        for index in range(count)
    ]


def test_free_running_and_teacher_forced_metrics_are_separated() -> None:
    model = Controlled()
    model.answers = torch.stack([torch.randint(0, vocab, (2, 4)) for vocab in TINY_VOCABS], dim=-1)
    batch = {"latents": torch.randn(2, 12, 128), "pool": torch.rand(2, 4, 12), "target": model.answers}
    metrics = evaluate(model, [batch], device=torch.device("cpu"))
    assert metrics["semantic_top1"] == 1.0
    assert metrics["teacher_forced_semantic_top1"] == 1.0
    assert metrics["teacher_forced_acoustic_top1"] == 1.0
    assert metrics["acoustic_top1"] == 0.0
    assert metrics["book7_top1"] == 0.0
    assert metrics["teacher_forced_book7_top5"] == 1.0
    assert metrics["acoustic_top5"] == 1.0
    assert metrics["kl"] == 0.0
    assert metrics["loss"] > 0


def test_ignored_frames_do_not_count() -> None:
    model = Controlled()
    model.answers = torch.stack([torch.randint(0, vocab, (2, 4)) for vocab in TINY_VOCABS], dim=-1)
    target = model.answers.clone()
    target[0] = IGNORE
    batch = {"latents": torch.randn(2, 12, 128), "pool": torch.rand(2, 4, 12), "target": target}
    metrics = evaluate(model, [batch], device=torch.device("cpu"))
    assert metrics["semantic_top1"] == 1.0
    assert metrics["teacher_forced_acoustic_top1"] == 1.0


def test_max_batches_and_reduce_hook() -> None:
    model = RvqEncoder(tiny_config()).eval()
    calls: list[int] = []

    def reduce(values: Tensor) -> Tensor:
        calls.append(int(values[3].item()))
        return values * 2

    metrics = evaluate(model, batches(model, 3), device=torch.device("cpu"), max_batches=2, reduce=reduce)
    assert calls == [16]
    assert metrics["loss"] > 0


def test_counter_metrics_layout() -> None:
    counter = Counter.zeros(2)
    counter.add_loss(torch.tensor(2.0), torch.tensor(1.5), torch.tensor(0.5), 4)
    counter.add_hits(2, 0, torch.tensor([1, 2]), torch.tensor([4, 4]), forced=True)
    metrics = counter.metrics(2, depth=False)
    assert metrics["loss"] == 2.0 and metrics["ce"] == 1.5 and metrics["kl"] == 0.5
    assert metrics["semantic_top1"] == 0.25
    assert metrics["acoustic_top1"] == 0.5
    assert "teacher_forced_semantic_top1" not in metrics
