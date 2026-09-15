from itertools import pairwise

import pytest
import torch

from rvq_ae.alignment import (
    Chunk,
    Pool,
    nominal_bounds,
    pool_matrix,
    stitched_bounds,
)
from rvq_ae.constants import STITCH_HOP


def spans(bounds: list[int]) -> list[int]:
    return [end - start for start, end in pairwise(bounds)]


def test_nominal_bounds_follow_the_stitched_hop() -> None:
    bounds = nominal_bounds(4500)
    assert len(bounds) == 4501
    assert bounds[0] == 0
    assert bounds[100] == STITCH_HOP
    assert bounds[200] == 2 * STITCH_HOP
    assert bounds[4500] == 15524


TWO_CHUNKS = (Chunk(0, 0, 200, 0, 431), Chunk(1, 100, 300, 431, 776))


def test_stitched_bounds_use_owned_spans() -> None:
    bounds = stitched_bounds(225, TWO_CHUNKS)
    assert bounds[0] == 0
    assert bounds[125] == 431
    assert bounds[225] == 776
    assert min(spans(bounds)) >= 0


def test_pool_matrix_averages_each_span() -> None:
    pool = pool_matrix([10, 12, 15, 16])
    assert pool.shape == (3, 6)
    assert torch.allclose(pool.sum(dim=1), torch.ones(3))
    assert torch.allclose(pool[0, :2], torch.full((2,), 0.5))
    assert torch.allclose(pool[1, 2:5], torch.full((3,), 1 / 3))
    assert pool[2, 5] == 1.0


@pytest.mark.parametrize("frames", [4, 37, 128, 1000])
def test_segment_pooling_equals_the_dense_operator(frames: int) -> None:
    """Pool.apply is the same linear map as the published matrix P, to machine precision."""
    bounds = nominal_bounds(frames)
    dense = pool_matrix(bounds, torch.float64)
    hidden = torch.randn(1, dense.shape[1], 16, dtype=torch.float64)
    assert torch.allclose(torch.bmm(dense[None], hidden), Pool.of(bounds).batched().apply(hidden), atol=1e-14)


def test_segment_pooling_ignores_right_padding() -> None:
    """Latents past the last boundary land on the sink row and reach no frame."""
    bounds = nominal_bounds(8)
    covered = bounds[-1]
    pool = Pool.of(bounds, length=covered + 5).batched()
    hidden = torch.randn(1, covered + 5, 16, dtype=torch.float64)
    hidden[:, covered:] = 1e9
    dense = pool_matrix(bounds, torch.float64)
    assert torch.allclose(torch.bmm(dense[None], hidden[:, :covered]), pool.apply(hidden), atol=1e-14)
