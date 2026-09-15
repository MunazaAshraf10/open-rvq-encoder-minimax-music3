import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from conftest import tiny_dav
from rvq_ae.dav import DavEncoder, Snake, fold_weight_norm


def legacy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict in the released layout: every conv stored as weight_g and weight_v."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            dims = tuple(range(1, module.weight.ndim))
            state[f"{name}.weight_g"] = module.weight.norm(dim=dims, keepdim=True)
            state[f"{name}.weight_v"] = module.weight.detach().clone()
            state[f"{name}.bias"] = module.bias.detach().clone()
        elif isinstance(module, Snake):
            state[f"{name}.alpha"] = module.alpha.detach().clone()
    return state


def test_folding_matches_torch_weight_norm() -> None:
    conv = weight_norm(nn.Conv1d(3, 5, kernel_size=7), name="weight", dim=0)
    gain = conv.parametrizations.weight.original0
    direction = conv.parametrizations.weight.original1
    folded = fold_weight_norm(
        {"c.weight_g": gain.detach(), "c.weight_v": direction.detach(), "c.bias": conv.bias.detach()}
    )
    assert set(folded) == {"c.weight", "c.bias"}
    assert torch.allclose(folded["c.weight"], conv.weight.detach(), atol=1e-6)


def latent_frames_for(samples: int, hop: int) -> int:
    return -(-samples // hop)


def test_default_geometry_matches_the_released_encoder() -> None:
    with torch.device("meta"):
        dav = DavEncoder()
    assert dav.hop == 512
    assert dav.latent_channels == 128
    stem = dav.encoder.block[0]
    assert stem.in_channels == 1 and stem.out_channels == 64
    assert dav.mean_proj.in_channels == 1024


def test_segmented_encoding_matches_a_single_pass() -> None:
    dav = tiny_dav().eval()
    audio = torch.randn(1, 2, 8 * 300 + 5)
    single = dav(audio)
    segmented = dav.encode(audio, segment_seconds=8 * 40 / 44_100, overlap_seconds=8 * 40 / 44_100)
    assert segmented.shape == single.shape
    assert torch.allclose(segmented, single, atol=1e-5)
    short = dav.encode(
        audio[:, :, : 8 * 20], segment_seconds=8 * 40 / 44_100, overlap_seconds=8 * 40 / 44_100
    )
    assert torch.allclose(short, dav(audio[:, :, : 8 * 20]))
