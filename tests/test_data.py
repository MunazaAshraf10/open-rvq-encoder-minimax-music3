import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from conftest import TINY_VOCABS, synthetic_cache, tiny_dav, write_shard
from rvq_ae.alignment import Pool, nominal_bounds
from rvq_ae.constants import CACHE_FORMAT, IGNORE
from rvq_ae.data.cache import build_cache, cache_paths, cached, read_meta
from rvq_ae.data.dataset import WindowDataset, check_codes, collate, crop_start
from rvq_ae.data.records import load_records, parse_entry, read_index


def test_index_parsing_reads_the_job_manifest(tmp_path: Path) -> None:
    entry = write_shard(tmp_path, shard_id=7, frames=40, split="Holdout")
    record = parse_entry(entry)
    assert record is not None
    assert record.shard_id == 7
    assert record.split == "holdout"
    assert record.stem == "shard-000007-sample-7"
    assert record.offset == 1
    assert record.exact
    assert record.chunks is not None and record.chunks[0].latent_end == nominal_bounds(40)[-1]
    assert record.vocab_sizes == TINY_VOCABS


def test_failed_jobs_and_wrong_codebook_counts_are_rejected(tmp_path: Path) -> None:
    entry = write_shard(tmp_path, shard_id=1)
    entry["manifest"]["jobs"][0]["status"] = "failed"
    assert parse_entry(entry) is None
    entry["manifest"]["jobs"][0]["status"] = "succeeded"
    entry["manifest"]["jobs"][0]["codebook_vocab_sizes"] = [4, 4]
    with pytest.raises(ValueError, match="codebooks"):
        parse_entry(entry)


def test_load_records_groups_by_split(tmp_path: Path) -> None:
    for shard, split in enumerate(("train", "train", "holdout")):
        write_shard(tmp_path, shard_id=shard, split=split)
    grouped = load_records(tmp_path)
    assert sorted(grouped) == ["holdout", "train"]
    assert [record.shard_id for record in grouped["train"]] == [0, 1]
    assert len(read_index(tmp_path / "indexes" / "batch-000000.jsonl")) == 3
    assert load_records(tmp_path / "indexes", limit=1)["train"][0].shard_id == 0


def test_cache_writes_latents_codes_and_teacher_tensors(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1)
    record = load_records(corpus)["train"][0]
    meta = read_meta(root, record)
    assert meta["format"] == CACHE_FORMAT
    assert meta["latent_frames"] == -(-44_100 // 8)
    assert meta["code_frames"] == 26
    assert meta["has_topk"] and meta["topk"] == 3
    assert meta["exact"]
    assert meta["mapped_latent_frames"] == nominal_bounds(25)[-1]
    assert cached(root, record, need_topk=True)

    with safe_open(cache_paths(root, record)[0], framework="pt") as handle:
        assert set(handle.keys()) == {"latents", "codes", "teacher_topk_ids", "teacher_topk_logits"}
        assert handle.get_slice("latents").get_shape() == [meta["latent_frames"], 8]


def test_cache_is_skipped_when_present_and_split_across_ranks(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=3)
    records = [r for group in load_records(corpus).values() for r in group]
    assert (
        build_cache(
            records, root=root, dav=tiny_dav(), device=torch.device("cpu"), corpus=corpus, progress=False
        )
        == 0
    )
    other = tmp_path / "other"
    counts = [
        build_cache(
            records,
            root=other,
            dav=tiny_dav(),
            device=torch.device("cpu"),
            corpus=corpus,
            rank=rank,
            world=2,
            progress=False,
        )
        for rank in range(2)
    ]
    assert counts == [2, 1]


def test_cache_without_teacher_tensors_is_not_enough_for_kl(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1, topk=0)
    record = load_records(corpus)["train"][0]
    assert cached(root, record, need_topk=False)
    assert not cached(root, record, need_topk=True)
    with pytest.raises(ValueError, match="teacher top k"):
        WindowDataset([record], root, size=8, need_topk=True)


def test_dataset_applies_the_priming_offset(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1)
    record = load_records(corpus)["train"][0]
    dataset = WindowDataset([record], root, size=8, stride=8, need_topk=True)
    assert len(dataset) == (25 - 8) // 8 + 1
    sample = dataset[1]

    stored = load_file(cache_paths(root, record)[0])
    assert torch.equal(sample["target"], stored["codes"][9:17].long())
    assert torch.equal(sample["ids"], stored["teacher_topk_ids"][9:17].long())
    bounds = nominal_bounds(25)
    assert sample["latents"].shape == (bounds[16] - bounds[8], 8)
    assert torch.allclose(sample["latents"], stored["latents"][bounds[8] : bounds[16]].float())
    assert sample["pool"].frame.shape == (bounds[16] - bounds[8],)
    assert torch.equal(sample["pool"].span.sum(), torch.tensor(float(bounds[16] - bounds[8])))


def test_random_crop_is_deterministic_per_epoch(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1)
    record = load_records(corpus)["train"][0]
    dataset = WindowDataset([record], root, size=8, stride=8, random_crop=True, seed=3)
    first = dataset[0]["target"]
    assert torch.equal(first, dataset[0]["target"])
    dataset.set_epoch(1)
    starts = {crop_start(3, epoch, 0, 17) for epoch in range(20)}
    assert len(starts) > 1
    assert all(0 <= start <= 17 for start in starts)
    assert dataset[0]["target"].shape == (8, 8)


def test_exact_only_filters_nominal_records(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_shard(corpus, shard_id=0, exact=True)
    write_shard(corpus, shard_id=1, exact=False)
    root = tmp_path / "cache"
    records = load_records(corpus)["train"]
    build_cache(records, root=root, dav=tiny_dav(), device=torch.device("cpu"), corpus=corpus, progress=False)
    assert len(WindowDataset(records, root, size=8, stride=8)) == 3
    assert len(WindowDataset(records, root, size=8, stride=8, exact_only=False)) == 6


def test_short_tracks_are_dropped(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1)
    record = load_records(corpus)["train"][0]
    assert len(WindowDataset([record], root, size=26)) == 0


def test_collate_pads_latents_and_sends_padding_to_the_sink_row() -> None:
    a = {
        "latents": torch.ones(4, 3),
        "pool": Pool.of([0, 2, 4]),
        "target": torch.zeros(2, 8, dtype=torch.long),
    }
    b = {
        "latents": torch.ones(6, 3),
        "pool": Pool.of([0, 3, 6]),
        "target": torch.ones(2, 8, dtype=torch.long),
    }
    batch = collate([a, b])
    assert batch["latents"].shape == (2, 6, 3)
    assert torch.all(batch["latents"][0, 4:] == 0)
    assert batch["pool"].frame.shape == (2, 6)
    assert batch["pool"].span.shape == (2, 2)
    # the two padded columns of the shorter window point at frame 2, the discarded sink row
    assert batch["pool"].frame[0].tolist() == [0, 0, 1, 1, 2, 2]
    assert batch["pool"].frame[1].tolist() == [0, 0, 0, 1, 1, 1]
    assert batch["target"].shape == (2, 2, 8)
    assert "ids" not in batch


def test_collate_padding_does_not_reach_any_frame() -> None:
    """A short window padded into a wider batch pools exactly as it does on its own."""
    short = {
        "latents": torch.randn(4, 3),
        "pool": Pool.of([0, 2, 4]),
        "target": torch.zeros(2, 8, dtype=torch.long),
    }
    long = {
        "latents": torch.randn(9, 3),
        "pool": Pool.of([0, 4, 9]),
        "target": torch.zeros(2, 8, dtype=torch.long),
    }
    alone = short["pool"].batched().apply(short["latents"][None])
    batch = collate([short, long])
    padded = batch["pool"].apply(batch["latents"])[:1]
    assert torch.allclose(alone, padded)


def test_collate_rejects_mixed_teacher_presence() -> None:
    a = {
        "latents": torch.ones(4, 3),
        "pool": Pool.of([0, 2, 4]),
        "target": torch.zeros(2, 8, dtype=torch.long),
    }
    b = dict(a, ids=torch.zeros(2, 8, 3, dtype=torch.long), teacher=torch.zeros(2, 8, 3))
    with pytest.raises(ValueError, match="mixed batch"):
        collate([a, b])
    batch = collate([b, b])
    assert batch["ids"].shape == (2, 2, 8, 3)


def test_code_validation() -> None:
    codes = torch.tensor([[0, 4], [IGNORE, 2]])
    check_codes(codes, (17, 5), "ok")
    with pytest.raises(ValueError, match="codebook 1"):
        check_codes(torch.tensor([[0, 5]]), (17, 5), "bad")
    with pytest.raises(ValueError, match="codebook 0"):
        check_codes(torch.tensor([[-3, 1]]), (17, 5), "bad")


def test_meta_sidecar_is_json(tmp_path: Path) -> None:
    corpus, root = synthetic_cache(tmp_path, shards=1)
    record = load_records(corpus)["train"][0]
    text = cache_paths(root, record)[1].read_text()
    assert json.loads(text)["record"] == record.stem
