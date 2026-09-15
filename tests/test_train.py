import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from conftest import synthetic_cache, tiny_config
from rvq_ae.schedule import cosine_lambda
from rvq_ae.train import TrainConfig, train


def config(tmp_path: Path, corpus: Path, root: Path, **overrides: object) -> TrainConfig:
    values: dict[str, object] = {
        "output": str(tmp_path / "run"),
        "model": tiny_config(latent_channels=8, mup_readout_zero_init=False).to_dict(),
        "dataset": str(corpus),
        "corpus": str(corpus),
        "cache": str(root),
        "window": 8,
        "stride": 8,
        "batch_size": 2,
        "workers": 0,
        "epochs": 10,
        "max_steps": 4,
        "warmup": 1,
        "lr": 1e-3,
        "precision": "fp32",
        "device": "cpu",
        "log_every": 1,
        "validate_every": 2,
        "checkpoint_every": 2,
    }
    values.update(overrides)
    return TrainConfig.from_dict(values)


def read_metrics(output: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def test_resume_reproduces_an_uninterrupted_run_bitwise(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path)
    full = train(config(tmp_path, corpus, root, output=str(tmp_path / "full"), validate_every=0))
    resumed = train(
        config(
            tmp_path,
            corpus,
            root,
            output=str(tmp_path / "resumed"),
            resume=str(full / "checkpoint-2"),
            validate_every=0,
        )
    )
    reference = load_file(full / "final" / "rvq_encoder.safetensors")
    continued = load_file(resumed / "final" / "rvq_encoder.safetensors")
    assert all(torch.equal(reference[key], continued[key]) for key in reference)
    full_rows = {row["step"]: row["train/loss"] for row in read_metrics(full) if "train/loss" in row}
    resumed_rows = {row["step"]: row["train/loss"] for row in read_metrics(resumed) if "train/loss" in row}
    assert set(resumed_rows) == {3, 4}
    assert all(full_rows[step] == resumed_rows[step] for step in resumed_rows)
    state = json.loads((resumed / "final" / "trainer_state.json").read_text())
    assert state["epoch"] == 1


def test_cosine_schedule_reproduces_the_published_v1_trace() -> None:
    """Half period 500: one at step zero, the floor at 500, 1,500, ..., 17,500, no warm up."""
    kwargs = {"half_period": 500, "floor": 1e-7 / 3e-4}
    assert cosine_lambda(0, **kwargs) == 1.0
    for minimum in range(500, 18_000, 1_000):
        assert cosine_lambda(minimum, **kwargs) == pytest.approx(kwargs["floor"], abs=1e-12)
        assert cosine_lambda(minimum + 500, **kwargs) == pytest.approx(1.0, abs=1e-12)
    assert cosine_lambda(250, **kwargs) == pytest.approx(0.5 + kwargs["floor"] / 2, abs=1e-6)
    with pytest.raises(ValueError, match="half_period"):
        cosine_lambda(1, half_period=0, floor=0.0)
