import json
from pathlib import Path

import torch

from conftest import synthetic_cache, tiny_config
from rvq_ae.cli import build_parser, checkpoints, main
from rvq_ae.train import TrainConfig, train


def test_parser_exposes_every_command() -> None:
    parser = build_parser()
    argv = {
        "cache": ["cache"],
        "train": ["train", "--config", "configs/v4_169m.json"],
        "evaluate": ["evaluate", "--run", "output"],
        "encode": ["encode", "--audio", "clip.flac", "--topk", "5"],
        "push": ["push", "--folder", "output/final", "--repo", "user/repo"],
        "serve": ["serve", "--port", "9000"],
    }
    for command, args in argv.items():
        parsed = parser.parse_args(args)
        assert parsed.command == command
        assert parsed.func.__name__ == f"cmd_{command}"
    assert parser.parse_args(["serve"]).device == "auto"
    assert parser.parse_args(["encode", "--audio", "a.wav"]).device == "cuda"


def test_train_then_evaluate_command(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path)
    cfg = TrainConfig.from_dict(
        {
            "output": str(tmp_path / "run"),
            "model": tiny_config(latent_channels=8).to_dict(),
            "dataset": str(corpus),
            "corpus": str(corpus),
            "cache": str(root),
            "window": 8,
            "stride": 8,
            "batch_size": 2,
            "workers": 0,
            "max_steps": 2,
            "warmup": 1,
            "precision": "fp32",
            "device": "cpu",
            "validate_every": 0,
            "checkpoint_every": 1,
        }
    )
    run = train(cfg)
    found = checkpoints(run)
    assert [item[0] for item in found] == ["checkpoint-1", "best/checkpoint-2", "checkpoint-2", "final"]
    assert [item[1] for item in found] == [1, 2, 2, 2]
    main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(corpus),
            "--cache",
            str(root),
            "--window",
            "8",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    rows = json.loads((run / "evaluation" / "metrics.json").read_text())
    assert [row["checkpoint"] for row in rows] == [
        "checkpoint-1",
        "best/checkpoint-2",
        "checkpoint-2",
        "final",
    ]
    assert rows[1]["loss"] == rows[2]["loss"] == rows[3]["loss"]
    assert (run / "evaluation" / "metrics.csv").read_text().startswith("checkpoint,step,loss")


def write_tiny_dav(folder: Path) -> Path:
    from safetensors.torch import save_file

    from conftest import tiny_dav
    from test_dav import legacy_state

    (folder / "audio_vae").mkdir(parents=True)
    save_file(legacy_state(tiny_dav()), folder / "audio_vae" / "diffusion_pytorch_model.safetensors")
    config = {
        "encoder_dim": 4,
        "encoder_rates": [2, 4],
        "encoder_latent_dim": 8,
        "channel_latent_channels": 4,
        "sampling_rate": 44100,
    }
    (folder / "audio_vae" / "config.json").write_text(json.dumps(config))
    return folder


def test_cache_and_encode_commands(tmp_path: Path) -> None:
    import soundfile
    from safetensors.torch import load_file

    from conftest import write_shard
    from rvq_ae.hub import save_encoder
    from rvq_ae.model import RvqEncoder

    corpus = tmp_path / "corpus"
    write_shard(corpus, shard_id=0)
    write_shard(corpus, shard_id=1, split="holdout")
    dav = write_tiny_dav(tmp_path / "dav")
    cache = tmp_path / "cache"
    main(
        [
            "cache",
            "--dataset",
            str(corpus),
            "--corpus",
            str(corpus),
            "--cache",
            str(cache),
            "--dav",
            str(dav),
            "--device",
            "cpu",
            "--split",
            "train",
        ]
    )
    assert len(list(cache.rglob("*.safetensors"))) == 1
    main(
        [
            "cache",
            "--dataset",
            str(corpus),
            "--corpus",
            str(corpus),
            "--cache",
            str(cache),
            "--dav",
            str(dav),
            "--device",
            "cpu",
        ]
    )
    assert len(list(cache.rglob("*.safetensors"))) == 2

    save_encoder(RvqEncoder(tiny_config(latent_channels=8)), tmp_path / "model")
    clip = tmp_path / "clip.wav"
    soundfile.write(clip, torch.randn(44_100, 2).numpy() * 0.1, 44_100)
    out = tmp_path / "codes.safetensors"
    main(
        [
            "encode",
            "--model",
            str(tmp_path / "model"),
            "--dav",
            str(dav),
            "--audio",
            str(clip),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--topk",
            "2",
        ]
    )
    tensors = load_file(out)
    assert tensors["codes"].shape == (25, 8)
    assert tensors["candidates"].shape == (25, 8, 2)
