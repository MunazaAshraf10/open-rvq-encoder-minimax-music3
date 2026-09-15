import torch
import torch.nn.functional as F

from conftest import published_config, tiny_config
from rvq_ae.model import RvqEncoder
from rvq_ae.mup import Readout, no_decay, param_groups, wide


def test_readout_divides_its_input_by_the_width_multiplier() -> None:
    readout = Readout(32, 7, 4.0, output_mult=1.5)
    x = torch.randn(3, 32)
    expected = F.linear(1.5 * x / 4.0, readout.weight, readout.bias)
    assert torch.allclose(readout(x), expected)
    assert set(readout.state_dict()) == {"weight", "bias"}


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
