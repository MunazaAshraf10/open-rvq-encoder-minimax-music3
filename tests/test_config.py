import json
from pathlib import Path

import pytest

from conftest import DATA, published_config
from rvq_ae.config import EncoderConfig


@pytest.mark.parametrize("tag", ["v1", "v2", "v4"])
def test_published_config_round_trips_byte_identical(tag: str) -> None:
    raw = (DATA / f"{tag}_config.json").read_text()
    cfg = EncoderConfig.from_dict(json.loads(raw))
    assert json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n" == raw


def test_v4_width_multiplier_and_attention_scale() -> None:
    cfg = published_config("v4")
    assert cfg.width_mult == 1088 / 128
    assert cfg.head_dim == 64
    assert cfg.attention_scale == 8.0 / 64


def test_v1_has_no_depth_decoder() -> None:
    cfg = published_config("v1")
    assert not cfg.depth_decoder
    assert cfg.width_mult == 4.0


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        EncoderConfig.from_dict({"d_model": 64, "mert_layer": 9})


def test_invalid_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EncoderConfig(d_model=64, num_heads=3)


def test_base_width_is_written_only_when_custom(tmp_path: Path) -> None:
    cfg = EncoderConfig(mup=True, mup_base_width=64)
    path = tmp_path / "config.json"
    cfg.save(path)
    values = json.loads(path.read_text())
    assert values["mup_base_width"] == 64
    assert EncoderConfig.load(path) == cfg
    assert "mup_base_width" not in EncoderConfig(mup=True).to_dict()


def test_standard_parametrisation_has_unit_multiplier() -> None:
    cfg = EncoderConfig(mup=False, d_model=256, num_heads=4)
    assert cfg.width_mult == 1.0
    assert cfg.attention_scale == 64**-0.5
