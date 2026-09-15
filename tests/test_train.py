import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from conftest import synthetic_cache, tiny_config
from rvq_ae.schedule import cosine_lambda, lr_lambda, make_scheduler
from rvq_ae.train import Dist, TrainConfig, train


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


def test_two_epochs_of_training_write_checkpoints_and_metrics(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path)
    cfg = config(tmp_path, corpus, root)
    output = train(cfg)
    assert (output / "final" / "rvq_encoder.safetensors").is_file()
    assert (output / "checkpoint-2" / "training_state.pt").is_file()
    assert (output / "checkpoint-4").is_dir()
    assert (output / "train_config.json").is_file()
    rows = read_metrics(output)
    train_rows = [row for row in rows if "train/loss" in row]
    validation_rows = [row for row in rows if "validation/loss" in row]
    assert [row["step"] for row in train_rows] == [1, 2, 3, 4]
    assert [row["step"] for row in validation_rows] == [2, 4, 4]
    assert "validation/teacher_forced_acoustic_top1" in validation_rows[0]
    assert "validation/book7_top5" in validation_rows[0]
    assert train_rows[-1]["train/epoch"] == 1
    assert train_rows[0]["train/kl"] > 0
    best = sorted((output / "best").iterdir())
    assert best and best[0].name.startswith("checkpoint-")
    state = json.loads((output / "final" / "trainer_state.json").read_text())
    assert state["step"] == 4 and state["config"]["max_steps"] == 4


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


def test_kl_training_requires_teacher_tensors_in_the_cache(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, topk=0)
    with pytest.raises(ValueError, match="teacher top k"):
        train(config(tmp_path, corpus, root))
    output = train(
        config(tmp_path, corpus, root, kl_weight=0.0, max_steps=1, validate_every=0, checkpoint_every=0)
    )
    assert read_metrics(output)[0]["train/kl"] == 0.0


def test_train_config_round_trip_and_unknown_keys(tmp_path: Path) -> None:
    cfg = TrainConfig(betas=(0.8, 0.9))
    path = tmp_path / "train.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert TrainConfig.load(path, lr=5e-4).lr == 5e-4
    assert TrainConfig.load(path).betas == (0.8, 0.9)
    with pytest.raises(ValueError, match="unknown training keys"):
        TrainConfig.from_dict({"learning_rate": 1e-3})


def test_single_process_dist_defaults() -> None:
    world = Dist.init("cpu")
    assert world.main and world.world == 1
    values = torch.arange(3, dtype=torch.float64)
    assert torch.equal(world.reduce(values), values)
    assert world.gather("x") == ["x"]


def test_schedule_shape() -> None:
    kwargs = {"warmup": 10, "total": 110, "floor": 0.1}
    assert lr_lambda(0, **kwargs) == 0.0
    assert lr_lambda(5, **kwargs) == 0.5
    assert lr_lambda(10, **kwargs) == 1.0
    assert lr_lambda(60, **kwargs) == pytest.approx(0.55)
    assert lr_lambda(110, **kwargs) == pytest.approx(0.1)
    assert lr_lambda(500, **kwargs) == pytest.approx(0.1)
    assert lr_lambda(3, warmup=0, total=3, floor=0.0) == 0.0


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


def test_make_scheduler_dispatches_on_the_schedule_name() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    linear = make_scheduler(optimizer, schedule="linear", warmup=10, total=100, floor=0.0)
    assert linear.get_last_lr()[0] == 0.0
    cosine = make_scheduler(optimizer, schedule="cosine", warmup=10, total=100, floor=0.0)
    assert cosine.get_last_lr()[0] == 1.0
    with pytest.raises(ValueError, match="unknown schedule"):
        make_scheduler(optimizer, schedule="sine", warmup=10, total=100, floor=0.0)
    with pytest.raises(ValueError, match="unknown schedule"):
        TrainConfig.from_dict({"schedule": "sine"})
