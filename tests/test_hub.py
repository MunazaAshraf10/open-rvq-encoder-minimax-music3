from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from conftest import tiny_config
from rvq_ae.constants import LEGACY_WEIGHTS_FORMAT
from rvq_ae.hub import VARIANTS, export_state, load_encoder, relative_files, resolve_encoder, save_encoder
from rvq_ae.model import RvqEncoder


def test_save_and_load_round_trip_strictly(tmp_path: Path) -> None:
    model = RvqEncoder(tiny_config(mup_readout_zero_init=False))
    files = save_encoder(model, tmp_path / "final")
    assert files.config.name == "rvq_encoder_config.json"
    assert files.weights.name == "rvq_encoder.safetensors"
    loaded = load_encoder(tmp_path / "final")
    assert loaded.cfg == model.cfg
    assert all(
        torch.equal(a, b)
        for a, b in zip(loaded.state_dict().values(), model.state_dict().values(), strict=True)
    )
    assert not any(param.requires_grad for param in loaded.parameters())
    assert not loaded.training


def test_export_strips_the_ddp_prefix() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    wrapped = torch.nn.Module()
    wrapped.module = model
    assert set(export_state(wrapped)) == {"0.weight", "0.bias"}
    assert all(value.dtype == torch.float32 for value in export_state(wrapped).values())


def test_subfolder_layout(tmp_path: Path) -> None:
    model = RvqEncoder(tiny_config())
    save_encoder(model, tmp_path / "best" / "checkpoint-7")
    files = resolve_encoder(tmp_path, subfolder="best/checkpoint-7")
    assert files.weights == tmp_path / "best" / "checkpoint-7" / "rvq_encoder.safetensors"


def test_release_layout(tmp_path: Path) -> None:
    model = RvqEncoder(tiny_config())
    stem = tmp_path / VARIANTS["v4"]
    stem.parent.mkdir()
    model.cfg.save(stem.with_suffix(".json"))
    save_file(
        export_state(model), stem.with_suffix(".safetensors"), metadata={"format": LEGACY_WEIGHTS_FORMAT}
    )
    loaded = load_encoder(tmp_path, variant="v4")
    assert loaded.cfg == model.cfg


def test_unknown_variant_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        relative_files("v9", None)
    with pytest.raises(FileNotFoundError):
        resolve_encoder(tmp_path)


def test_wrong_format_tag_is_rejected(tmp_path: Path) -> None:
    model = RvqEncoder(tiny_config())
    model.cfg.save(tmp_path / "rvq_encoder_config.json")
    save_file(export_state(model), tmp_path / "rvq_encoder.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="format"):
        load_encoder(tmp_path)
