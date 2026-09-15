import torch
import torch.nn.functional as F

from conftest import published_config, tiny_config
from rvq_ae.layers import attention
from rvq_ae.model import RvqEncoder
from rvq_ae.mup import Readout, no_decay, param_groups, wide, wide_fan_in


def test_readout_divides_its_input_by_the_width_multiplier() -> None:
    readout = Readout(32, 7, 4.0, output_mult=1.5)
    x = torch.randn(3, 32)
    expected = F.linear(1.5 * x / 4.0, readout.weight, readout.bias)
    assert torch.allclose(readout(x), expected)
    assert set(readout.state_dict()) == {"weight", "bias"}


def test_zero_init_readout_gives_a_uniform_distribution() -> None:
    readout = Readout(32, 7, 4.0, zero_init=True)
    assert torch.count_nonzero(readout.weight) == 0
    assert torch.allclose(torch.softmax(readout(torch.randn(2, 32)), dim=-1), torch.full((2, 7), 1 / 7))


def test_v4_matrix_like_parameters_are_the_54_width_squared_weights() -> None:
    with torch.device("meta"):
        model = RvqEncoder(published_config("v4"))
    names = [name for name in dict(model.named_parameters()) if wide(name)]
    assert len(names) == 8 * 6 + 3 * 2
    assert all(name.startswith(("transformer.", "blocks.")) for name in names)
    assert not any(wide(name) for name in dict(model.named_parameters()) if name.startswith("depth_decoder"))
    assert not wide("conv_in.weight")
    assert not wide("heads.0.weight")
    assert not wide("position")


def test_fan_in_predicate_covers_readouts_but_not_the_stem() -> None:
    assert wide_fan_in("heads.0.bias")
    assert wide_fan_in("transformer.3.linear2.bias")
    assert wide_fan_in("depth_decoder.context_projection.weight")
    assert not wide_fan_in("conv_in.bias")
    assert not wide_fan_in("depth_decoder.heads.0.bias")


def test_no_decay_names() -> None:
    assert no_decay("transformer.0.norm1.weight")
    assert no_decay("depth_decoder.prior_embeddings.0.weight")
    assert no_decay("conv_in.bias")
    assert not no_decay("position")
    assert not no_decay("transformer.0.q_proj.weight")


def test_param_groups_scale_learning_rate_and_decay_of_matrix_like_weights() -> None:
    cfg = tiny_config()
    model = RvqEncoder(cfg)
    groups = param_groups(model.named_parameters(), lr=1e-3, weight_decay=0.01, width_mult=cfg.width_mult)
    by_param = {id(param): group for group in groups for param in group["params"]}
    for name, param in model.named_parameters():
        group = by_param[id(param)]
        if wide(name):
            assert group["lr"] == 1e-3 / 2.0
            assert group["weight_decay"] == 0.01 * 2.0
        else:
            assert group["lr"] == 1e-3
            assert group["weight_decay"] == (0.0 if no_decay(name) else 0.01)
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


def test_init_rescales_biases_and_nonzero_readouts_by_sqrt_width_mult() -> None:
    torch.manual_seed(1)
    standard = RvqEncoder(tiny_config(mup=False, mup_readout_zero_init=False))
    torch.manual_seed(1)
    scaled = RvqEncoder(tiny_config(mup=True, mup_readout_zero_init=False))
    root = 2.0**0.5

    def ratio(name: str) -> torch.Tensor:
        return dict(scaled.named_parameters())[name] / dict(standard.named_parameters())[name]

    assert torch.allclose(ratio("transformer.0.q_proj.bias"), torch.full((32,), root))
    assert torch.allclose(ratio("transformer.0.linear2.bias"), torch.full((32,), root))
    assert torch.allclose(ratio("blocks.1.conv1.bias"), torch.full((32,), root))
    assert torch.allclose(ratio("heads.0.bias"), torch.full((17,), root))
    assert torch.allclose(ratio("heads.0.weight"), torch.full((17, 32), root))
    assert torch.allclose(ratio("depth_decoder.context_projection.weight"), torch.full((16, 32), root))
    assert torch.allclose(ratio("conv_in.bias"), torch.ones(32))
    assert torch.allclose(ratio("transformer.0.q_proj.weight"), torch.ones(32, 32))
    assert torch.allclose(ratio("depth_decoder.heads.0.bias"), torch.ones(5))


def test_zero_init_readout_is_not_rescaled() -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=True))
    assert torch.count_nonzero(model.heads[0].weight) == 0


def test_attention_scale_is_one_over_head_dim_under_mup() -> None:
    cfg = tiny_config()
    assert cfg.head_dim == 16
    assert cfg.attention_scale == 0.5
    assert tiny_config(mup=False).attention_scale == 0.25
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    fast = attention(q, k, v, scale=0.5, causal=False, dropout=0.0, training=False)
    slow = attention(q, k, v, scale=0.25, causal=False, dropout=0.0, training=False)
    assert not torch.allclose(fast, slow)


def test_initial_logit_scale_is_stable_across_widths() -> None:
    latents = torch.randn(4, 8, 128)
    pool = torch.eye(8).unsqueeze(0).repeat(4, 1, 1)
    stds = []
    for width in (32, 64, 128):
        torch.manual_seed(3)
        model = RvqEncoder(
            tiny_config(d_model=width, num_heads=width // 16, mup_readout_zero_init=False)
        ).eval()
        with torch.no_grad():
            stds.append(model(latents, pool)[0].std().item())
    assert max(stds) / min(stds) < 2.0
