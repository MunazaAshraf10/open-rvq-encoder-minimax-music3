import pytest
import torch
from torch import Tensor

from conftest import tiny_config
from rvq_ae.alignment import Pool, nominal_bounds
from rvq_ae.layers import Layer, attention, attention_reference
from rvq_ae.model import RvqEncoder

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# (name, dtype, tolerance): bfloat16 carries eight mantissa bits, so agreement to 1e-2 on
# activations of unit scale is the resolution of the format rather than a numerical defect.
DTYPES = [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)]


def qkv(
    device: str, dtype: torch.dtype, length: int = 128, heads: int = 17, dim: int = 64
) -> tuple[Tensor, Tensor, Tensor]:
    """Three independent [2, heads, length, dim] tensors on the given device."""
    shape = (2, heads, length, dim)
    generator = torch.Generator(device=device).manual_seed(0)
    drawn = [torch.randn(shape, generator=generator, device=device, dtype=dtype) for index in range(3)]
    return drawn[0], drawn[1], drawn[2]


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_fused_attention_matches_the_reference(dtype: torch.dtype, tol: float, causal: bool) -> None:
    q, k, v = qkv("cpu", dtype)
    kwargs = {"scale": 0.125, "causal": causal, "dropout": 0.0, "training": False}
    assert torch.allclose(attention(q, k, v, **kwargs), attention_reference(q, k, v, **kwargs), atol=tol)


@CUDA
@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_fused_attention_matches_the_reference_on_cuda(dtype: torch.dtype, tol: float, causal: bool) -> None:
    q, k, v = qkv("cuda", dtype)
    kwargs = {"scale": 0.125, "causal": causal, "dropout": 0.0, "training": False}
    assert torch.allclose(attention(q, k, v, **kwargs), attention_reference(q, k, v, **kwargs), atol=tol)


def test_attention_honours_the_mup_scale() -> None:
    """The scale reaches the kernel: two different scales cannot give the same output."""
    q, k, v = qkv("cpu", torch.float32, length=8, heads=2, dim=16)
    kwargs = {"causal": False, "dropout": 0.0, "training": False}
    wide = attention(q, k, v, scale=0.5, **kwargs)
    narrow = attention(q, k, v, scale=0.25, **kwargs)
    assert not torch.allclose(wide, narrow)


def test_causal_attention_cannot_see_the_future() -> None:
    """Position 0 of a causal layer attends to itself alone, whatever follows it."""
    q, k, v = qkv("cpu", torch.float32, length=8, heads=2, dim=16)
    kwargs = {"scale": 0.25, "causal": True, "dropout": 0.0, "training": False}
    changed = v.clone()
    changed[:, :, 1:] += 100.0
    assert torch.allclose(attention(q, k, v, **kwargs)[:, :, 0], attention(q, k, changed, **kwargs)[:, :, 0])


def test_dropout_is_inactive_outside_training() -> None:
    q, k, v = qkv("cpu", torch.float32, length=8, heads=2, dim=16)
    kwargs = {"scale": 0.25, "causal": False, "dropout": 0.5}
    assert torch.equal(
        attention(q, k, v, training=False, **kwargs), attention(q, k, v, training=False, **kwargs)
    )


def test_layer_rejects_a_width_that_heads_do_not_divide() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Layer(30, 4, 2, 0.0, scale=0.25, causal=False)


@CUDA
def test_encoder_logits_agree_between_cpu_and_cuda() -> None:
    """The float32 forward pass carries no device specific path, only float reassociation.

    Logits are the invariant worth asserting. Codes are not: argmax over a 16,384 way readout
    flips wherever two candidates sit within float noise of each other, which on real audio is
    common enough to make an exact code comparison a flaky test rather than a meaningful one.
    The readout sums a full width of products, so the measured gap is a few 1e-4 in float32;
    1e-3 is a thousand times tighter than bfloat16 resolution and above that noise.
    """
    torch.manual_seed(0)
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False)).eval()
    bounds = nominal_bounds(8)
    latents = torch.randn(1, bounds[-1], 128)
    pool = Pool.of(bounds).batched()
    with torch.no_grad():
        host = model(latents, pool)
        device = model.cuda()(latents.cuda(), pool.to(torch.device("cuda")))
    for book, (a, b) in enumerate(zip(host, device, strict=True)):
        assert torch.allclose(a, b.cpu(), atol=1e-3), book
