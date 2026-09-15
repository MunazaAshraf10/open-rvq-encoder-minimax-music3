import json
from pathlib import Path

import pytest
import torch

from conftest import TINY_VOCABS, published_config, published_keys, tiny_config
from rvq_ae.alignment import Pool
from rvq_ae.config import EncoderConfig
from rvq_ae.constants import IGNORE
from rvq_ae.model import RvqEncoder

ROOT = Path(__file__).resolve().parents[1]


def inputs(batch: int = 2, frames: int = 4, latents_per_frame: int = 3) -> tuple[torch.Tensor, Pool]:
    """Latents [B, L, 128] and the pooling operator for L latents split evenly over frames."""
    length = frames * latents_per_frame
    bounds = list(range(0, length + 1, latents_per_frame))
    single = Pool.of(bounds)
    pool = Pool(frame=single.frame.expand(batch, -1), span=single.span.expand(batch, -1))
    return torch.randn(batch, length, 128), pool


def random_codes(batch: int = 2, frames: int = 4) -> torch.Tensor:
    return torch.stack([torch.randint(0, vocab, (batch, frames)) for vocab in TINY_VOCABS], dim=-1)


@pytest.mark.parametrize(
    ("tag", "count", "keys"), [("v1", 40_978_944, 167), ("v2", 154_736_064, 167), ("v4", 169_008_576, 210)]
)
def test_published_parameter_counts(tag: str, count: int, keys: int) -> None:
    with torch.device("meta"):
        model = RvqEncoder(published_config(tag))
    assert model.parameter_count() == count
    assert len(model.state_dict()) == keys


@pytest.mark.parametrize("tag", ["v1", "v4"])
def test_state_dict_keys_match_the_released_checkpoints(tag: str) -> None:
    with torch.device("meta"):
        model = RvqEncoder(published_config(tag))
    assert set(model.state_dict()) == published_keys(tag)


def test_logit_shapes_follow_codebook_vocabularies(cfg: EncoderConfig) -> None:
    model = RvqEncoder(cfg).eval()
    latents, pool = inputs()
    for scores, vocab in zip(model(latents, pool), TINY_VOCABS, strict=True):
        assert scores.shape == (2, 4, vocab)
    for scores, vocab in zip(model(latents, pool, random_codes()), TINY_VOCABS, strict=True):
        assert scores.shape == (2, 4, vocab)


def test_independent_heads_when_the_depth_decoder_is_off() -> None:
    model = RvqEncoder(tiny_config(depth_decoder=False)).eval()
    assert model.depth_decoder is None
    assert len(model.heads) == 8
    latents, pool = inputs()
    plain = model(latents, pool)
    forced = model(latents, pool, random_codes())
    assert all(torch.equal(a, b) for a, b in zip(plain, forced, strict=True))


def test_depth_decoder_conditions_only_on_earlier_codebooks(cfg: EncoderConfig) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).eval()
    latents, pool = inputs()
    codes = random_codes()
    base = model(latents, pool, codes)
    changed = codes.clone()
    changed[:, :, 3] = (changed[:, :, 3] + 1) % 5
    other = model(latents, pool, changed)
    for book in range(8):
        same = torch.allclose(base[book], other[book])
        assert same == (book <= 3), f"codebook {book}"


def test_free_running_equals_teacher_forcing_on_its_own_greedy_codes(cfg: EncoderConfig) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).eval()
    latents, pool = inputs()
    free = model(latents, pool)
    greedy = torch.stack([scores.argmax(dim=-1) for scores in free], dim=-1)
    forced = model(latents, pool, greedy)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(free, forced, strict=True))
    assert torch.equal(model.codes(latents, pool), greedy)


def test_ignored_targets_embed_as_token_zero(cfg: EncoderConfig) -> None:
    model = RvqEncoder(cfg).eval()
    latents, pool = inputs()
    codes = random_codes()
    masked = codes.clone()
    masked[:, :, :] = IGNORE
    zeros = torch.zeros_like(codes)
    assert all(
        torch.equal(a, b)
        for a, b in zip(model(latents, pool, masked), model(latents, pool, zeros), strict=True)
    )


def test_every_parameter_receives_a_gradient(cfg: EncoderConfig) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).train()
    latents, pool = inputs()
    loss = sum(scores.float().logsumexp(-1).mean() for scores in model(latents, pool, random_codes()))
    loss.backward()
    missing = [name for name, param in model.named_parameters() if param.grad is None]
    assert missing == []


def test_context_longer_than_positions_is_rejected(cfg: EncoderConfig) -> None:
    model = RvqEncoder(cfg)
    latents, pool = inputs(frames=9)
    with pytest.raises(ValueError, match="frame context"):
        model(latents, pool)


def test_gradient_checkpointing_matches_the_plain_forward(cfg: EncoderConfig) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).train()
    latents, pool = inputs()
    codes = random_codes()
    plain = model(latents, pool, codes)
    model.checkpointing = True
    checked = model(latents, pool, codes)
    assert all(torch.allclose(a, b) for a, b in zip(plain, checked, strict=True))


def test_compiled_model_matches_eager(cfg: EncoderConfig) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).eval()
    latents, pool = inputs()
    eager = model(latents, pool, random_codes() * 0)
    try:
        compiled = torch.compile(model)(latents, pool, torch.zeros(2, 4, 8, dtype=torch.long))
    except RuntimeError as error:
        pytest.skip(f"torch.compile unavailable here: {error}")
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(eager, compiled, strict=True))


@pytest.mark.parametrize("tag", ["v1", "v2", "v4"])
def test_published_provenance_counts_match_the_code(tag: str) -> None:
    """The parameter counts recorded with the released weights are the counts this code builds."""
    path = ROOT / "results" / "published" / "experiment-summary.json"
    if not path.is_file():
        pytest.skip("published results are not checked in")
    summary = json.loads(path.read_text())
    recorded = {entry["version"]: entry["parameter_count"] for entry in summary["encoders"]}
    with torch.device("meta"):
        model = RvqEncoder(published_config(tag))
    assert model.parameter_count() == recorded[tag]
