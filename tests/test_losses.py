import pytest
import torch
import torch.nn.functional as F

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


def test_cross_entropy_is_the_per_codebook_mean() -> None:
    logits, target = make()[:2]
    ce = codebook_ce(logits, target)
    for book, scores in enumerate(logits):
        expected = F.cross_entropy(scores.flatten(0, 1), target[:, :, book].flatten())
        assert torch.allclose(ce[book], expected)


def test_cross_entropy_skips_ignored_frames() -> None:
    logits, target = make()[:2]
    target[0, :, 0] = IGNORE
    ce = codebook_ce(logits, target)
    expected = F.cross_entropy(logits[0][1].float(), target[1, :, 0])
    assert torch.allclose(ce[0], expected)


def test_total_loss_is_ce_plus_weighted_kl() -> None:
    logits, target, ids, teacher = make()
    loss = rvq_loss(logits, target, ids=ids, teacher=teacher, kl_weight=0.25, tau=1.0)
    ce = codebook_ce(logits, target).mean()
    kl = topk_kl(logits, ids, teacher, target, tau=1.0).mean()
    assert torch.allclose(loss.total, ce + 0.25 * kl)
    assert loss.kl.item() > 0
    loss.total.backward()
    assert all(scores.grad is not None for scores in logits)


def test_kl_without_weight_ignores_teacher_tensors() -> None:
    logits, target = make()[:2]
    loss = rvq_loss(logits, target)
    assert torch.equal(loss.total, loss.ce)
    assert loss.kl.item() == 0.0


def test_kl_weight_requires_teacher_tensors() -> None:
    logits, target = make()[:2]
    with pytest.raises(ValueError, match="teacher"):
        rvq_loss(logits, target, kl_weight=0.25)


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


def test_kl_is_zero_when_the_student_matches_the_teacher() -> None:
    logits = [torch.zeros(1, 1, 4)]
    logits[0][0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ids = torch.tensor([[[[0, 1, 2, 3]]]])
    teacher = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    assert torch.allclose(topk_kl(logits, ids, teacher, target, tau=1.0), torch.zeros(1), atol=1e-6)


def test_kl_excludes_end_of_track_and_negative_ids() -> None:
    logits = [torch.randn(1, 2, 5)]
    ids = torch.tensor([[[[0, 5]], [[-1, 5]]]])
    teacher = torch.zeros(1, 2, 1, 2)
    target = torch.zeros(1, 2, 1, dtype=torch.long)
    kl = topk_kl(logits, ids, teacher, target, tau=1.0)[0]
    # frame 1 has no valid id, frame 0 keeps only id 0 with teacher probability 1
    expected = -torch.log_softmax(logits[0][0, 0], dim=-1)[0]
    assert torch.allclose(kl, expected)


def test_kl_with_no_valid_frame_is_zero_but_differentiable() -> None:
    logits = [torch.randn(1, 2, 5, requires_grad=True)]
    ids = torch.full((1, 2, 1, 3), 5)
    teacher = torch.zeros(1, 2, 1, 3)
    target = torch.zeros(1, 2, 1, dtype=torch.long)
    kl = topk_kl(logits, ids, teacher, target, tau=1.0)
    assert kl.item() == 0.0
    assert kl.requires_grad


def test_kl_skips_ignored_targets() -> None:
    logits, target, ids, teacher = make(batch=1, frames=2)
    target[0, 1, :] = IGNORE
    full = topk_kl(logits, ids, teacher, target.masked_fill(target == IGNORE, 0), tau=1.0)
    masked = topk_kl(logits, ids, teacher, target, tau=1.0)
    first_only = topk_kl([s[:, :1] for s in logits], ids[:, :1], teacher[:, :1], target[:, :1], tau=1.0)
    assert torch.allclose(masked, first_only)
    assert not torch.allclose(masked, full)


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
