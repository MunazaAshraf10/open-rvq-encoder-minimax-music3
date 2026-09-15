import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rvq_ae.alignment import Pool, nominal_bounds, pool_matrix
from rvq_ae.bench.harness import Report, Result, free_memory, measure
from rvq_ae.config import EncoderConfig
from rvq_ae.constants import COLLECTION, DAV_REPO, FRAME_RATE, LATENT_CHANNELS, SAMPLE_RATE, WINDOW
from rvq_ae.inference import CodeEncoder
from rvq_ae.layers import attention, attention_reference
from rvq_ae.losses import rvq_loss
from rvq_ae.model import DepthDecoder, RvqEncoder
from rvq_ae.mup import param_groups

CONFIGS = Path("configs")
PUBLISHED_STEPS = 17_660

# The v4 temporal encoder runs 17 heads of 64 over a 128 frame window. The depth decoder runs
# 8 heads of 64 over a depth of 8, once per frame, so its batch is batch * frames.
SHAPES = {
    "temporal (b16, 17h, len 128)": (16, 17, WINDOW, 64, False),
    "depth (b2048, 8h, len 8)": (2048, 8, 8, 64, True),
}


def train_config(name: str) -> dict[str, Any]:
    return dict(json.loads((CONFIGS / name).read_text(encoding="utf-8")))


def model_config(name: str) -> EncoderConfig:
    return EncoderConfig.from_dict(train_config(name)["model"])


def window_pool(batch: int, frames: int, device: torch.device) -> tuple[Tensor, Pool]:
    """A batch of identical windows: latents [B, L, 128] and the pooling operator over them."""
    bounds = nominal_bounds(frames)
    single = Pool.of(bounds)
    pool = Pool(frame=single.frame.expand(batch, -1), span=single.span.expand(batch, -1)).to(device)
    return torch.randn(batch, bounds[-1], LATENT_CHANNELS, device=device), pool


def random_codes(cfg: EncoderConfig, batch: int, frames: int, device: torch.device) -> Tensor:
    return torch.stack(
        [torch.randint(0, vocab, (batch, frames), device=device) for vocab in cfg.codebook_vocab_sizes],
        dim=-1,
    )


def teacher_topk(
    cfg: EncoderConfig, batch: int, frames: int, device: torch.device, topk: int = 50
) -> tuple[Tensor, Tensor]:
    """Stored generator top k ids and logits [B, T, K, k], the distillation target of the objective."""
    ids = torch.stack(
        [torch.randint(0, vocab, (batch, frames, topk), device=device) for vocab in cfg.codebook_vocab_sizes],
        dim=2,
    )
    return ids, torch.randn(batch, frames, len(cfg.codebook_vocab_sizes), topk, device=device)


def attend_forward(
    fn: Callable[..., Tensor], q: Tensor, k: Tensor, v: Tensor, scale: float, causal: bool
) -> None:
    with torch.no_grad():
        fn(q, k, v, scale=scale, causal=causal, dropout=0.0, training=False)


def attend_backward(
    fn: Callable[..., Tensor], q: Tensor, k: Tensor, v: Tensor, scale: float, causal: bool
) -> None:
    for tensor in (q, k, v):
        tensor.grad = None
    out = fn(q, k, v, scale=scale, causal=causal, dropout=0.0, training=False)
    torch.autograd.backward(out.sum())


def attention_suite(report: Report, device: torch.device, dtype: torch.dtype) -> None:
    """Fused scaled dot product attention against the unfused reference, forward and backward."""
    for case, (batch, heads, length, dim, causal) in SHAPES.items():
        shape = (batch, heads, length, dim)
        q, k, v = (torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for index in range(3))
        scale = dim**-0.5 if causal else 8.0 / dim
        with torch.no_grad():
            fused = attention(q, k, v, scale=scale, causal=causal, dropout=0.0, training=False)
            plain = attention_reference(q, k, v, scale=scale, causal=causal, dropout=0.0, training=False)
            error = (fused - plain).abs().max()
        for variant, fn in (("reference", attention_reference), ("fused sdpa", attention)):
            for phase, body in (("forward", attend_forward), ("backward", attend_backward)):
                free_memory(device)
                ms, peak = measure(partial(body, fn, q, k, v, scale, causal), device)
                report.add(
                    Result(
                        suite="attention",
                        case=f"{case} {phase}",
                        variant=variant,
                        milliseconds=ms,
                        memory=peak,
                        note=f"max abs difference {error:.1e}" if variant == "fused sdpa" else "",
                    )
                )
        free_memory(device)


def pooling_suite(report: Report, device: torch.device) -> None:
    """The dense pooling matrix against the segment mean that computes the same operator."""
    for batch, frames in ((16, WINDOW), (64, WINDOW), (16, 512)):
        bounds = nominal_bounds(frames)
        length = bounds[-1]
        hidden = torch.randn(batch, length, 1088, device=device)
        dense = pool_matrix(bounds).to(device).expand(batch, -1, -1)
        single = Pool.of(bounds)
        pool = Pool(frame=single.frame.expand(batch, -1), span=single.span.expand(batch, -1)).to(device)
        with torch.no_grad():
            error = (torch.bmm(dense, hidden) - pool.apply(hidden)).abs().max()
        for variant, run in (
            ("dense bmm", partial(torch.bmm, dense, hidden)),
            ("segment mean", partial(pool.apply, hidden)),
        ):
            free_memory(device)
            ms, peak = measure(run, device, reps=50)
            report.add(
                Result(
                    suite="pooling",
                    case=f"b{batch} x {frames} frames x {length} latents x 1088",
                    variant=variant,
                    milliseconds=ms,
                    memory=peak,
                    # the dense path goes through a tf32 matmul while the segment path accumulates
                    # in float32, so this difference is the baseline's rounding, not the new path's
                    note=f"max abs difference {error:.1e}" if variant == "segment mean" else "",
                )
            )
        free_memory(device)
    collate_suite(report)


def build_dense(bounds: list[int], batch: int, frames: int, length: int) -> Tensor:
    """What collate used to do: one dense [T, L] matrix per sample, padded into [B, T, L]."""
    out = torch.zeros(batch, frames, length)
    for row in range(batch):
        matrix = pool_matrix(bounds)
        out[row, :, : matrix.shape[1]] = matrix
    return out


def build_segment(bounds: list[int], batch: int, frames: int, length: int) -> Pool:
    """What collate does now: one index vector per sample, padded into [B, L]."""
    frame = torch.full((batch, length), frames, dtype=torch.int64)
    spans = []
    for row in range(batch):
        pool = Pool.of(bounds)
        frame[row, : pool.frame.shape[0]] = pool.frame
        spans.append(pool.span)
    return Pool(frame=frame, span=torch.stack(spans))


def collate_suite(report: Report) -> None:
    """Host side cost of assembling the pooling operator for a batch, which the dataloader pays."""
    host = torch.device("cpu")
    for batch, frames in ((16, WINDOW), (64, WINDOW)):
        bounds = nominal_bounds(frames)
        length = bounds[-1]
        case = f"collate b{batch} x {frames} frames x {length} latents"
        for variant, run in (
            ("dense matrix", partial(build_dense, bounds, batch, frames, length)),
            ("segment index", partial(build_segment, bounds, batch, frames, length)),
        ):
            ms, peak = measure(run, host, warmup=3, reps=20)
            elements = batch * frames * length if variant == "dense matrix" else batch * length
            report.add(
                Result(
                    suite="pooling",
                    case=case,
                    variant=variant,
                    milliseconds=ms,
                    memory=peak,
                    note=f"{elements:,} elements built per batch",
                )
            )


def encoder_step(
    model: RvqEncoder,
    latents: Tensor,
    pool: Pool,
    targets: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    train: bool,
) -> None:
    with torch.no_grad() if not train else torch.enable_grad():
        with torch.autocast(device.type, dtype=dtype):
            logits = model(latents, pool, targets)
            if not train:
                return
            loss = rvq_loss(logits, targets, kl_weight=0.0)
    model.zero_grad(set_to_none=True)
    torch.autograd.backward(loss.total)


def encoder_suite(report: Report, device: torch.device, dtype: torch.dtype) -> None:
    """Forward and training step throughput of the released widths, in frames per second."""
    for name, tag in (("v1_41m.json", "v1 41M"), ("v4_169m.json", "v4 169M")):
        cfg = model_config(name)
        model = RvqEncoder(cfg).to(device)
        for batch in (8, 16, 32):
            latents, pool = window_pool(batch, WINDOW, device)
            targets = random_codes(cfg, batch, WINDOW, device)
            for variant, checkpointing, train in (
                ("forward, teacher forced", False, False),
                ("forward and backward", False, True),
                ("forward and backward, checkpointed", True, True),
            ):
                model.checkpointing = checkpointing
                model.train(train)
                free_memory(device)
                run = partial(encoder_step, model, latents, pool, targets, device, dtype, train=train)
                try:
                    ms, peak = measure(run, device, warmup=3, reps=10)
                except torch.OutOfMemoryError:
                    free_memory(device)
                    continue
                report.add(
                    Result(
                        suite="encoder",
                        case=f"{tag} batch {batch}",
                        variant=variant,
                        milliseconds=ms,
                        memory=peak,
                        throughput=batch * WINDOW / (ms / 1000.0),
                        units="frames/s",
                    )
                )
            free_memory(device)
        free_memory(device)


def depth_step(
    decoder: DepthDecoder,
    features: Tensor,
    argument: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    forced: bool,
) -> None:
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        if forced:
            decoder(features, argument)
        else:
            decoder.generate(features, argument)


def depth_suite(report: Report, device: torch.device, dtype: torch.dtype) -> None:
    """Teacher forced depth decoding (one pass) against free running (seven sequential steps)."""
    cfg = model_config("v4_169m.json")
    model = RvqEncoder(cfg).to(device).eval()
    decoder = model.depth_decoder
    if decoder is None:
        raise ValueError("the v4 configuration must carry a depth decoder")
    batch = 16
    latents, pool = window_pool(batch, WINDOW, device)
    targets = random_codes(cfg, batch, WINDOW, device)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        features = model.features(latents, pool)
    for variant, argument, forced in (
        ("teacher forced", targets, True),
        ("free running", targets[:, :, 0], False),
    ):
        free_memory(device)
        run = partial(depth_step, decoder, features, argument, device, dtype, forced=forced)
        ms, peak = measure(run, device, warmup=3, reps=10)
        report.add(
            Result(
                suite="depth decoder",
                case=f"v4 169M batch {batch} x {WINDOW} frames",
                variant=variant,
                milliseconds=ms,
                memory=peak,
                throughput=batch * WINDOW / (ms / 1000.0),
                units="frames/s",
            )
        )
    free_memory(device)


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    latents: Tensor,
    pool: Pool,
    targets: Tensor,
    teacher: tuple[Tensor, Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    accumulation: int,
    kl_weight: float,
) -> None:
    ids, logits = teacher
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accumulation):
        with torch.autocast(device.type, dtype=dtype):
            loss = rvq_loss(
                model(latents, pool, targets), targets, ids=ids, teacher=logits, kl_weight=kl_weight
            )
        torch.autograd.backward(loss.total / accumulation)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def training_suite(report: Report, device: torch.device, dtype: torch.dtype) -> None:
    """One full optimizer step at the released single GPU recipe, accumulation included."""
    name, tag = "v4_169m_single_gpu.json", "v4 169M single GPU"
    values = train_config(name)
    cfg = model_config(name)
    batch = int(values.get("batch_size", 16))
    accumulation = int(values.get("accumulation", 1))
    model = RvqEncoder(cfg).to(device)
    model.checkpointing = bool(values.get("gradient_checkpointing", False))
    model.train()
    optimizer = torch.optim.AdamW(
        param_groups(
            model.named_parameters(),
            lr=float(values.get("lr", 3e-4)),
            weight_decay=float(values.get("weight_decay", 0.01)),
            width_mult=cfg.width_mult,
        ),
        fused=device.type == "cuda",
    )
    latents, pool = window_pool(batch, WINDOW, device)
    targets = random_codes(cfg, batch, WINDOW, device)
    kl_weight = float(values.get("kl_weight", 0.25))
    teacher = teacher_topk(cfg, batch, WINDOW, device)
    free_memory(device)
    run = partial(
        training_step,
        model,
        optimizer,
        latents,
        pool,
        targets,
        teacher,
        device,
        dtype,
        accumulation=accumulation,
        kl_weight=kl_weight,
    )
    ms, peak = measure(run, device, warmup=2, reps=8)
    windows = batch * accumulation
    report.add(
        Result(
            suite="training",
            case=f"{tag} ({windows} windows per step)",
            variant=f"batch {batch} x accumulation {accumulation}",
            milliseconds=ms,
            memory=peak,
            throughput=windows * WINDOW / (ms / 1000.0),
            units="frames/s",
            note=f"{ms / 1000.0 * PUBLISHED_STEPS / 3600:.1f} h for the published {PUBLISHED_STEPS:,} steps",
        )
    )
    free_memory(device)


def audio_suite(report: Report, device: torch.device) -> None:
    """End to end real time factor: seconds of audio encoded per second of wall time."""

    codec = CodeEncoder.load(COLLECTION, variant="v4", dav=DAV_REPO, device=device)
    for seconds in (30.0, 180.0, 360.0):
        audio = torch.randn(2, int(seconds * SAMPLE_RATE))
        latents = codec.latents(audio, SAMPLE_RATE)
        frames = round(seconds * FRAME_RATE)
        for variant, run in (
            ("dav encoder", partial(codec.latents, audio, SAMPLE_RATE)),
            ("rvq encoder", partial(codec.encode_latents, latents, frames)),
            ("end to end", partial(codec.encode, audio, SAMPLE_RATE)),
        ):
            free_memory(device)
            ms, peak = measure(run, device, warmup=1, reps=3)
            report.add(
                Result(
                    suite="audio",
                    case=f"{seconds:.0f} s stereo track",
                    variant=variant,
                    milliseconds=ms,
                    memory=peak,
                    throughput=seconds / (ms / 1000.0),
                    units="x real time",
                )
            )
        free_memory(device)
