import torch

from rvq_ae.constants import IGNORE
from rvq_ae.losses import codebook_ce, rvq_loss, topk_hits, topk_kl

VOCABS = (7, 5, 5)


def make(
    batch: int = 2, frames: int = 3, k: int = 3
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = [torch.randn(batch, frames, vocab, requires_grad=True) for vocab in VOCABS]
    target = torch.stack([torch.randint(0, vocab, (batch, frames)) for vocab in VOCABS], dim=-1)
    ids = torch.stack([torch.randint(0, vocab, (batch, frames, k)) for vocab in VOCABS], dim=2)
    teacher = torch.randn(batch, frames, len(VOCABS), k)
    return logits, target, ids, teacher


def test_total_loss_is_ce_plus_weighted_kl() -> None:
    logits, target, ids, teacher = make()
    loss = rvq_loss(logits, target, ids=ids, teacher=teacher, kl_weight=0.25, tau=1.0)
    ce = codebook_ce(logits, target).mean()
    kl = topk_kl(logits, ids, teacher, target, tau=1.0).mean()
    assert torch.allclose(loss.total, ce + 0.25 * kl)
    assert loss.kl.item() > 0
    loss.total.backward()
    assert all(scores.grad is not None for scores in logits)


def test_kl_uses_the_teacher_distribution_on_its_support() -> None:
    logits, target, ids, teacher = make(batch=1, frames=1, k=2)
    logits = [logits[0]]
    ids = ids[:, :, :1]
    teacher = teacher[:, :, :1]
    target = target[:, :, :1]
    tau = 2.0
    kl = topk_kl(logits, ids, teacher, target, tau=tau)[0]
    prob_teacher = torch.softmax(teacher[0, 0, 0] / tau, dim=-1)
    log_student = torch.log_softmax(logits[0][0, 0] / tau, dim=-1)[ids[0, 0, 0]]
    expected = tau * tau * (prob_teacher * (prob_teacher.log() - log_student)).sum()
    assert torch.allclose(kl, expected)


def test_kl_excludes_end_of_track_and_negative_ids() -> None:
    logits = [torch.randn(1, 2, 5)]
    ids = torch.tensor([[[[0, 5]], [[-1, 5]]]])
    teacher = torch.zeros(1, 2, 1, 2)
    target = torch.zeros(1, 2, 1, dtype=torch.long)
    kl = topk_kl(logits, ids, teacher, target, tau=1.0)[0]
    # frame 1 has no valid id, frame 0 keeps only id 0 with teacher probability 1
    expected = -torch.log_softmax(logits[0][0, 0], dim=-1)[0]
    assert torch.allclose(kl, expected)


def test_topk_hits_masks_ignored_frames() -> None:
    logits = [torch.tensor([[[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]]])]
    target = torch.tensor([[[1], [IGNORE]]])
    correct, total = topk_hits(logits, target, 1)
    assert correct.tolist() == [1]
    assert total.tolist() == [1]
    target = torch.tensor([[[0], [2]]])
    correct, total = topk_hits(logits, target, 2)
    assert correct.tolist() == [1]
    assert total.tolist() == [2]
