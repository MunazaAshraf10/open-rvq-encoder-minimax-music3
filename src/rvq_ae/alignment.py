"""Timeline contract between DAV latents (86.13 Hz) and RVQ frames (25 Hz).

MiniMax Music 3 renders a track in 200 frame chunks with a 100 frame hop and stitches
the resulting DAV latents with a 345 latent hop. Frame t therefore maps to a variable
length latent span [s_t, e_t) rather than a fixed 3.4453125 latent stride. Every function
here returns or consumes the n + 1 boundaries s_0 <= s_1 <= ... <= s_n.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import Tensor

from rvq_ae.constants import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    HOP,
    OWNED_FROM,
    RATE_DEN,
    RATE_NUM,
    SAMPLE_RATE,
    STITCH_HOP,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True, slots=True)
class Chunk:
    """One rollout chunk: its frame span and the stitched latent span it produced."""

    index: int
    frame_start: int
    frame_end: int
    latent_start: int
    latent_end: int

    @classmethod
    def parse(cls, entry: Mapping[str, Any]) -> Self:
        """Build from a chunk_stitching entry of the dataset index."""
        return cls(
            index=int(entry["chunk_index"]),
            frame_start=int(entry["semantic_frame_start"]),
            frame_end=int(entry["semantic_frame_end_exclusive"]),
            latent_start=int(entry["stitched_flow_latent_start"]),
            latent_end=int(entry["stitched_flow_latent_end_exclusive"]),
        )


def nominal_bounds(frames: int) -> list[int]:
    """Reconstruct the stitched timeline for a track without chunk metadata.

    Chunk c starts at latent 345 c. Frame f belongs to chunk c = clamp((f - 25) // 100, 0, W - 1)
    with W = max(1, (n - 1) // 100) chunks. Inside chunk c with cf frames and cl = floor(cf 441 / 128)
    latents, local frame lf maps to latent ceil(lf cl / cf). The formula matches the generator for
    full chunks (100 frames -> 345 latents) and interpolates the shorter last chunk.
    """
    windows = max(1, (frames - 1) // CHUNK_HOP)
    bounds: list[int] = []
    for frame in range(frames + 1):
        chunk = min(max((frame - OWNED_FROM) // CHUNK_HOP, 0), windows - 1)
        local = frame - chunk * CHUNK_HOP
        chunk_frames = min(CHUNK_FRAMES, frames - chunk * CHUNK_HOP)
        chunk_latents = chunk_frames * RATE_NUM // RATE_DEN
        bounds.append(chunk * STITCH_HOP + ceil_div(local * chunk_latents, chunk_frames))
    return bounds


def stitched_bounds(frames: int, chunks: Sequence[Chunk]) -> list[int]:
    """Exact timeline from the generator's chunk metadata.

    Chunk 0 owns frames [f_0, f_1 + 25); chunk i > 0 owns [f_i + 25, f_(i+1) + 25); the last chunk
    owns through its own end, clamped to n. Owned frames map linearly (integer ceil) onto the kept
    latents of that chunk. A later chunk rewrites the boundary it shares with the previous one.
    """
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    bounds = [0] * (frames + 1)
    for position, chunk in enumerate(ordered):
        owner_start = chunk.frame_start if position == 0 else chunk.frame_start + OWNED_FROM
        if position + 1 < len(ordered):
            owner_end = ordered[position + 1].frame_start + OWNED_FROM
        else:
            owner_end = chunk.frame_end
        owner_start = min(owner_start, frames)
        owner_end = min(owner_end, frames)
        owned = owner_end - owner_start
        kept = chunk.latent_end - chunk.latent_start
        if owned <= 0 or kept <= 0:
            continue
        for frame in range(owner_start, owner_end + 1):
            bounds[frame] = chunk.latent_start + ceil_div((frame - owner_start) * kept, owned)
    for frame in range(frames):
        if bounds[frame + 1] < bounds[frame]:
            raise ValueError(f"chunk metadata is not monotonic at frame {frame}")
    return bounds


def frame_bounds(frames: int, chunks: Sequence[Chunk] | None) -> list[int]:
    """Exact timeline when chunk metadata is present, nominal reconstruction otherwise."""
    if chunks:
        return stitched_bounds(frames, chunks)
    return nominal_bounds(frames)


def usable_frames(frames: int, latent_frames: int, chunks: Sequence[Chunk] | None) -> int:
    """Largest frame count whose final boundary fits inside the available latents."""
    count = frames
    while count > 1 and frame_bounds(count, chunks)[-1] > latent_frames:
        count -= 1
    return count


def pool_matrix(bounds: Sequence[int]) -> Tensor:
    """Mean pooling operator P with P[t, l] = 1 / (e_t - s_t) for l in [s_t, e_t).

    Latent indices are local to bounds[0], so the caller slices latents [bounds[0], bounds[-1]).
    """
    if len(bounds) < 2:
        raise ValueError("at least two boundaries are required")
    origin = bounds[0]
    length = bounds[-1] - origin
    pool = torch.zeros(len(bounds) - 1, length)
    for frame in range(len(bounds) - 1):
        start = bounds[frame] - origin
        end = bounds[frame + 1] - origin
        if end <= start:
            raise ValueError(f"frame {frame} has an empty latent span [{start}, {end})")
        pool[frame, start:end] = 1.0 / (end - start)
    return pool


def frame_centers(bounds: Sequence[int]) -> Tensor:
    """Center time of every frame in seconds: (s_t + e_t) / 2 * hop / sample_rate."""
    edges = torch.tensor(bounds, dtype=torch.float64)
    return (edges[:-1] + edges[1:]) / 2 * HOP / SAMPLE_RATE
