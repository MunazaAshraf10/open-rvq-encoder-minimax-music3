"""Fixed length frame windows read from the latent cache."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import Dataset

from rvq_ae.alignment import frame_bounds, pool_matrix, usable_frames
from rvq_ae.constants import IGNORE, WINDOW
from rvq_ae.data.cache import cache_paths, read_meta
from rvq_ae.data.records import Record


class Sample(TypedDict):
    latents: Tensor
    pool: Tensor
    target: Tensor
    ids: NotRequired[Tensor]
    teacher: NotRequired[Tensor]


class Batch(TypedDict):
    latents: Tensor
    pool: Tensor
    target: Tensor
    ids: NotRequired[Tensor]
    teacher: NotRequired[Tensor]


@dataclass(frozen=True, slots=True)
class Window:
    record: int
    start: int
    usable: int


def crop_start(seed: int, epoch: int, index: int, span: int) -> int:
    """Deterministic crop offset in [0, span] from (seed, epoch, index)."""
    digest = hashlib.blake2b(f"{seed}:{epoch}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (span + 1)


def check_codes(codes: Tensor, vocabs: Sequence[int], stem: str) -> None:
    for book, vocab in enumerate(vocabs):
        column = codes[:, book]
        bad = ((column < 0) & (column != IGNORE)) | (column >= vocab)
        if bool(bad.any()):
            raise ValueError(f"{stem}: codebook {book} holds codes outside [0, {vocab})")


class WindowDataset(Dataset[Sample]):
    """Windows of size frames with the given stride; random_crop redraws the start every epoch."""

    def __init__(
        self,
        records: Sequence[Record],
        root: Path,
        *,
        size: int = WINDOW,
        stride: int = WINDOW,
        random_crop: bool = False,
        exact_only: bool = True,
        need_topk: bool = False,
        seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.root = root
        self.size = size
        self.stride = stride
        self.random_crop = random_crop
        self.need_topk = need_topk
        self.seed = seed
        self.epoch = 0
        self.bounds: dict[int, list[int]] = {}
        self.windows: list[Window] = []
        for index, record in enumerate(self.records):
            meta = read_meta(root, record)
            if exact_only and not record.exact:
                continue
            if need_topk and not meta["has_topk"]:
                raise ValueError(f"{record.stem}: teacher top k tensors are missing from the cache")
            frames = min(record.emitted_frames, int(meta["code_frames"]) - record.offset)
            usable = usable_frames(frames, int(meta["latent_frames"]), record.chunks)
            if usable < size:
                continue
            self.bounds[index] = frame_bounds(usable, record.chunks)
            for start in range(0, usable - size + 1, stride):
                self.windows.append(Window(index, start, usable))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Sample:
        window = self.windows[index]
        record = self.records[window.record]
        start = window.start
        if self.random_crop:
            start = crop_start(self.seed, self.epoch, index, window.usable - self.size)
        bounds = self.bounds[window.record][start : start + self.size + 1]
        rows = slice(start + record.offset, start + record.offset + self.size)
        tensors = cache_paths(self.root, record)[0]
        with safe_open(tensors, framework="pt", device="cpu") as handle:
            latents = handle.get_slice("latents")[bounds[0] : bounds[-1]].float()
            target = handle.get_slice("codes")[rows].long()
            sample: Sample = {"latents": latents, "pool": pool_matrix(bounds), "target": target}
            if self.need_topk:
                sample["ids"] = handle.get_slice("teacher_topk_ids")[rows].long()
                sample["teacher"] = handle.get_slice("teacher_topk_logits")[rows].float()
        check_codes(target, record.vocab_sizes, record.stem)
        return sample


def collate(samples: Sequence[Sample]) -> Batch:
    """Right pad latents and pool columns with zeros so every window keeps its exact spans."""
    length = max(sample["latents"].shape[0] for sample in samples)
    latents = torch.zeros(len(samples), length, samples[0]["latents"].shape[1])
    pool = torch.zeros(len(samples), samples[0]["pool"].shape[0], length)
    for row, sample in enumerate(samples):
        count = sample["latents"].shape[0]
        latents[row, :count] = sample["latents"]
        pool[row, :, :count] = sample["pool"]
    batch: Batch = {"latents": latents, "pool": pool, "target": torch.stack([s["target"] for s in samples])}
    with_topk = [("ids" in sample) for sample in samples]
    if any(with_topk):
        if not all(with_topk):
            raise ValueError("mixed batch: some windows carry teacher top k tensors and others do not")
        batch["ids"] = torch.stack([s["ids"] for s in samples])
        batch["teacher"] = torch.stack([s["teacher"] for s in samples])
    return batch
