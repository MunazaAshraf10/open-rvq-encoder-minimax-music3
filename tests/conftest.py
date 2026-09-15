import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch
from safetensors.torch import save as save_bytes

from rvq_ae.alignment import nominal_bounds
from rvq_ae.config import EncoderConfig
from rvq_ae.data.cache import build_cache
from rvq_ae.data.records import load_records
from rvq_ae.dav import DavEncoder

DATA = Path(__file__).parent / "data"

TINY_VOCABS = (17, 5, 5, 5, 5, 5, 5, 5)


def tiny_config(**overrides: object) -> EncoderConfig:
    """Two heads of dimension 16 so the muP attention scale differs from 1 / sqrt(d_head)."""
    values: dict[str, object] = {
        "codebook_vocab_sizes": TINY_VOCABS,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 2,
        "ff_mult": 2,
        "dropout": 0.0,
        "max_position_embeddings": 8,
        "mup": True,
        "mup_base_width": 16,
        "depth_decoder": True,
        "depth_decoder_dim": 16,
        "depth_decoder_layers": 1,
        "depth_decoder_heads": 2,
        "depth_decoder_ff_mult": 2,
        "depth_decoder_dropout": 0.0,
    }
    values.update(overrides)
    return EncoderConfig.from_dict(values)


@pytest.fixture
def cfg() -> EncoderConfig:
    return tiny_config()


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)


def published_config(tag: str) -> EncoderConfig:
    return EncoderConfig.from_dict(json.loads((DATA / f"{tag}_config.json").read_text()))


def published_keys(tag: str) -> set[str]:
    return set((DATA / f"{tag}_keys.txt").read_text().split())


def tiny_dav() -> DavEncoder:
    """Hop 8, four channel posterior per side, so latents have 8 channels."""
    return DavEncoder(dim=4, rates=(2, 4), latent_dim=8, out=4)


def write_shard(
    corpus: Path,
    *,
    shard_id: int,
    seconds: float = 1.0,
    frames: int = 25,
    split: str = "train",
    topk: int = 3,
    exact: bool = True,
    sample_rate: int = 44_100,
) -> dict[str, object]:
    """Write one synthetic shard ZIP and return its index entry."""

    rows = frames + 1
    codes = torch.stack([torch.randint(0, vocab, (rows,)) for vocab in TINY_VOCABS], dim=-1).to(torch.int16)
    tensors = {"codes": codes}
    if topk:
        tensors["teacher_topk_ids"] = torch.stack(
            [torch.randint(0, vocab, (rows, topk)) for vocab in TINY_VOCABS], dim=1
        ).to(torch.int32)
        tensors["teacher_topk_logits"] = torch.randn(rows, len(TINY_VOCABS), topk)
    audio = io.BytesIO()
    samples = int(seconds * sample_rate)
    soundfile.write(
        audio, np.random.default_rng(shard_id).standard_normal((samples, 2)) * 0.1, sample_rate, format="FLAC"
    )
    path = corpus / "data" / "00000" / f"shard-{shard_id:06d}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{shard_id}/audio.flac", audio.getvalue())
        archive.writestr(f"{shard_id}/prediction.safetensors", save_bytes(tensors))
    stitching = None
    if exact:
        stitching = [
            {
                "chunk_index": 0,
                "semantic_frame_start": 0,
                "semantic_frame_end_exclusive": frames,
                "stitched_flow_latent_start": 0,
                "stitched_flow_latent_end_exclusive": nominal_bounds(frames)[-1],
            }
        ]
    entry = {
        "shard_id": shard_id,
        "path": str(path.relative_to(corpus)),
        "manifest": {
            "jobs": [
                {
                    "id": f"sample-{shard_id}",
                    "status": "succeeded",
                    "audio_file": f"{shard_id}/audio.flac",
                    "tensor_file": f"{shard_id}/prediction.safetensors",
                    "emitted_frames": frames,
                    "sampling_rate": sample_rate,
                    "codebook_vocab_sizes": list(TINY_VOCABS),
                    "dataset_split": split,
                    "alignment": {"emitted_code_row_offset": 1, "chunk_stitching": stitching},
                }
            ]
        },
    }
    index = corpus / "indexes"
    index.mkdir(exist_ok=True)
    with (index / "batch-000000.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def synthetic_cache(
    tmp_path: Path, *, shards: int = 3, seconds: float = 1.0, topk: int = 3
) -> tuple[Path, Path]:
    """Corpus with a number of one second tracks (train, train, holdout) and its latent cache."""

    corpus = tmp_path / "corpus"
    for shard in range(shards):
        write_shard(
            corpus,
            shard_id=shard,
            seconds=seconds,
            split="holdout" if shards > 1 and shard == shards - 1 else "train",
            topk=topk,
        )
    root = tmp_path / "cache"
    records = [record for records in load_records(corpus).values() for record in records]
    build_cache(records, root=root, dav=tiny_dav(), device=torch.device("cpu"), corpus=corpus, progress=False)
    return corpus, root
