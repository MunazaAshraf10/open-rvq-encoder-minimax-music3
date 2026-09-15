import torch

from conftest import tiny_config, tiny_dav
from rvq_ae.alignment import Pool, nominal_bounds
from rvq_ae.inference import CodeEncoder, frame_count, frames_for_latents, windows
from rvq_ae.model import RvqEncoder


def encoder() -> CodeEncoder:
    cfg = tiny_config(latent_channels=8, mup_readout_zero_init=False)
    return CodeEncoder(tiny_dav(), RvqEncoder(cfg), device=torch.device("cpu"))


def test_windows_cover_every_frame_without_overlap() -> None:
    assert windows(300, 128) == [(0, 128), (128, 256), (256, 300)]
    assert windows(128, 128) == [(0, 128)]
    assert windows(1, 128) == [(0, 1)]


def test_frame_count_rounds_duration_and_respects_latents() -> None:
    assert frame_count(44100, 44100, 10_000) == 25
    assert frame_count(44100 * 400, 44100, 10**9) == 9000
    assert frame_count(44100, 44100, nominal_bounds(25)[-1] - 1) == 24
    assert frame_count(10, 44100, 10_000) == 1


def test_encode_produces_one_code_row_per_frame() -> None:
    codec = encoder()
    audio = torch.randn(2, 44100)
    result = codec.encode(audio, 44100, topk=3)
    assert result.frames == 25
    assert result.codes.shape == (25, 8)
    assert result.confidence.shape == (25, 8)
    assert result.candidates is not None and result.candidates.shape == (25, 8, 3)
    assert result.semantic_topk is not None and result.semantic_topk.shape == (25, 3)
    assert result.codes.dtype == torch.int64
    assert torch.all((result.confidence > 0) & (result.confidence <= 1))
    assert torch.all(result.codes[:, 0] < 17) and torch.all(result.codes[:, 1:] < 5)
    assert torch.equal(result.semantic_topk[:, 0], result.codes[:, 0])
    payload = result.to_dict()
    assert payload["duration"] == 1.0 and len(payload["codes"]) == 25


def test_windowed_encoding_matches_a_direct_window_forward() -> None:
    codec = encoder()
    frames = 20
    latents = torch.randn(nominal_bounds(frames)[-1], 8)
    result = codec.encode_latents(latents, frames)
    assert len(windows(frames, codec.window)) == 3
    bounds = nominal_bounds(frames)
    pool = Pool.of(bounds[8:17]).batched()
    direct = codec.model.codes(latents[bounds[8] : bounds[16]][None], pool)[0]
    assert torch.equal(result.codes[8:16], direct)


def test_resampling_keeps_the_frame_count() -> None:
    codec = encoder()
    audio = torch.randn(1, 24000)
    assert codec.encode(audio, 24000).frames == 25


def test_frames_for_latents_inverts_the_nominal_timeline() -> None:

    for frames in (1, 7, 25, 128, 300):
        latents = nominal_bounds(frames)[-1]
        assert nominal_bounds(frames_for_latents(latents))[-1] <= latents
        assert frames_for_latents(latents) >= frames - 1
    codec = encoder()
    latents = torch.randn(nominal_bounds(20)[-1], 8)
    assert codec.encode_latents(latents).frames == codec.encode_latents(latents, 20).frames
