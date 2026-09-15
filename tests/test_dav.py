import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from conftest import tiny_dav
from rvq_ae.dav import DavEncoder, Snake, fold_weight_norm, latent_frames, load_dav, read_weights


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


def test_snake_formula() -> None:
    snake = Snake(2)
    with torch.no_grad():
        snake.alpha[0, 1, 0] = 2.0
    x = torch.randn(1, 2, 5)
    expected = x + torch.sin(snake.alpha * x) ** 2 / snake.alpha
    assert torch.allclose(snake(x), expected, atol=1e-6)


def test_output_length_is_samples_over_hop() -> None:
    dav = tiny_dav().eval()
    assert dav.hop == 8
    for samples in (8, 9, 15, 16, 100):
        latents = dav(torch.randn(1, 2, samples))
        assert latents.shape == (1, 8, latent_frames_for(samples, dav.hop))
    assert latent_frames(1000) == 2


def latent_frames_for(samples: int, hop: int) -> int:
    return -(-samples // hop)


def test_stereo_channels_are_stacked_left_then_right() -> None:
    dav = tiny_dav().eval()
    stereo = torch.randn(1, 2, 64)
    latents = dav(stereo)
    left = dav(stereo[:, :1])
    right = dav(stereo[:, 1:])
    assert torch.allclose(latents[:, :4], left[:, :4])
    assert torch.allclose(latents[:, 4:], right[:, 4:])
    assert torch.allclose(left[:, :4], left[:, 4:])


def test_shape_handling() -> None:
    dav = tiny_dav().eval()
    assert dav(torch.randn(64)).shape == (1, 8, 8)
    assert dav(torch.randn(2, 64)).shape == (1, 8, 8)
    with pytest.raises(ValueError, match="mono or stereo"):
        dav(torch.randn(1, 3, 64))


def test_loader_reads_the_diffusers_layout_and_ignores_decoder_weights(tmp_path: Path) -> None:
    source = tiny_dav()
    state = legacy_state(source)
    state["decoder.model.0.weight_g"] = torch.ones(1, 1, 1)
    state["logs_proj.weight"] = torch.ones(4, 8, 1)
    state["dec_in_proj.bias"] = torch.ones(8)
    folder = tmp_path / "audio_vae"
    folder.mkdir()
    save_file(state, folder / "diffusion_pytorch_model.safetensors")
    config = {
        "encoder_dim": 4,
        "encoder_rates": [2, 4],
        "encoder_latent_dim": 8,
        "channel_latent_channels": 4,
        "sampling_rate": 44100,
    }
    (folder / "config.json").write_text(json.dumps(config))
    loaded = load_dav(tmp_path)
    audio = torch.randn(1, 2, 64)
    assert torch.allclose(loaded(audio), source.eval()(audio), atol=1e-5)
    assert not any(param.requires_grad for param in loaded.parameters())


def test_legacy_pth_reader_keeps_only_encoder_tensors(tmp_path: Path) -> None:
    source = tiny_dav()
    state = legacy_state(source)
    state["decoder.model.0.weight_g"] = torch.ones(1, 1, 1)
    state["flow.weight"] = torch.ones(1)
    path = tmp_path / "dav.pth"
    torch.save(state, path)
    folded = fold_weight_norm(read_weights(path))
    assert set(folded) == set(source.state_dict())
    assert all(torch.allclose(folded[key], value) for key, value in source.state_dict().items())


def test_loader_fails_on_missing_encoder_keys(tmp_path: Path) -> None:
    state = legacy_state(tiny_dav())
    del state["mean_proj.bias"]
    path = tmp_path / "dav.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match=r"mean_proj\.bias"):
        load_dav(path)


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
