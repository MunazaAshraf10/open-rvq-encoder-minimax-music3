from itertools import pairwise

import pytest
import torch

from rvq_ae.alignment import (
    Chunk,
    frame_bounds,
    frame_centers,
    nominal_bounds,
    pool_matrix,
    stitched_bounds,
    usable_frames,
)
from rvq_ae.constants import HOP, SAMPLE_RATE, STITCH_HOP


def spans(bounds: list[int]) -> list[int]:
    return [end - start for start, end in pairwise(bounds)]


def test_nominal_bounds_follow_the_stitched_hop() -> None:
    bounds = nominal_bounds(4500)
    assert len(bounds) == 4501
    assert bounds[0] == 0
    assert bounds[100] == STITCH_HOP
    assert bounds[200] == 2 * STITCH_HOP
    assert bounds[4500] == 15524


@pytest.mark.parametrize("frames", [1, 2, 25, 99, 100, 101, 200, 201, 333, 4500])
def test_nominal_bounds_are_monotonic_and_span_every_frame(frames: int) -> None:
    # every frame covers 3 or 4 latents (441 / 128 = 3.4453)
    assert set(spans(nominal_bounds(frames))) <= {3, 4}


def test_nominal_bounds_frames_before_warm_up_belong_to_chunk_zero() -> None:
    bounds = nominal_bounds(400)
    assert bounds[:26] == [(f * (200 * 441 // 128) + 199) // 200 for f in range(26)]


TWO_CHUNKS = (Chunk(0, 0, 200, 0, 431), Chunk(1, 100, 300, 431, 776))


def test_stitched_bounds_use_owned_spans() -> None:
    bounds = stitched_bounds(225, TWO_CHUNKS)
    assert bounds[0] == 0
    assert bounds[125] == 431
    assert bounds[225] == 776
    assert min(spans(bounds)) >= 0


def test_stitched_bounds_accept_unsorted_chunks() -> None:
    assert stitched_bounds(225, tuple(reversed(TWO_CHUNKS))) == stitched_bounds(225, TWO_CHUNKS)


def test_stitched_bounds_later_chunk_rewrites_the_shared_boundary() -> None:
    chunks = (Chunk(0, 0, 200, 0, 431), Chunk(1, 100, 300, 430, 776))
    assert stitched_bounds(225, chunks)[125] == 430


def test_stitched_bounds_reject_non_monotonic_metadata() -> None:
    chunks = (Chunk(0, 0, 200, 0, 431), Chunk(1, 100, 300, 100, 200))
    with pytest.raises(ValueError, match="monotonic"):
        stitched_bounds(225, chunks)


def test_frame_bounds_prefers_chunk_metadata() -> None:
    assert frame_bounds(225, TWO_CHUNKS) == stitched_bounds(225, TWO_CHUNKS)
    assert frame_bounds(225, None) == nominal_bounds(225)
    assert frame_bounds(225, ()) == nominal_bounds(225)


def test_chunk_parse_reads_dataset_field_names() -> None:
    chunk = Chunk.parse(
        {
            "chunk_index": 1,
            "semantic_frame_start": 100,
            "semantic_frame_end_exclusive": 300,
            "stitched_flow_latent_start": 431,
            "stitched_flow_latent_end_exclusive": 776,
        }
    )
    assert chunk == TWO_CHUNKS[1]


def test_usable_frames_shrinks_until_the_last_boundary_fits() -> None:
    assert usable_frames(300, 10_000, None) == 300
    assert usable_frames(300, nominal_bounds(300)[-1] - 1, None) == 299
    assert usable_frames(300, 0, None) == 1


def test_pool_matrix_averages_each_span() -> None:
    pool = pool_matrix([10, 12, 15, 16])
    assert pool.shape == (3, 6)
    assert torch.allclose(pool.sum(dim=1), torch.ones(3))
    assert torch.allclose(pool[0, :2], torch.full((2,), 0.5))
    assert torch.allclose(pool[1, 2:5], torch.full((3,), 1 / 3))
    assert pool[2, 5] == 1.0


def test_pool_matrix_rejects_empty_spans() -> None:
    with pytest.raises(ValueError, match="empty"):
        pool_matrix([0, 3, 3])
    with pytest.raises(ValueError, match="two boundaries"):
        pool_matrix([0])


def test_frame_centers_use_the_dav_hop() -> None:
    centers = frame_centers([0, 4, 7])
    expected = torch.tensor([2.0, 5.5], dtype=torch.float64) * HOP / SAMPLE_RATE
    assert torch.allclose(centers, expected)
