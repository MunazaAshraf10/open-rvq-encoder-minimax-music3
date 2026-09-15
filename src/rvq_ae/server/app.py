import io
import json
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import soundfile
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from safetensors.torch import load as load_bytes
from safetensors.torch import save as save_bytes
from torch import Tensor

from rvq_ae import __version__
from rvq_ae.audio import load_audio
from rvq_ae.constants import COLLECTION, DAV_REPO, FRAME_RATE, IGNORE
from rvq_ae.inference import CodeEncoder, EncodeResult
from rvq_ae.server.schemas import CodesBody, EncodeResponse, EvalResponse, ModelInfo

ENV_PREFIX = "RVQ_AE_"


@dataclass(frozen=True, slots=True)
class Settings:
    model: str = COLLECTION
    variant: str | None = "v4"
    subfolder: str | None = None
    dav: str = DAV_REPO
    revision: str | None = None
    device: str = "auto"
    max_seconds: float = 360.0

    @classmethod
    def from_env(cls) -> Self:
        """Read RVQ_AE_MODEL, RVQ_AE_VARIANT, RVQ_AE_SUBFOLDER, RVQ_AE_DAV, RVQ_AE_REVISION, RVQ_AE_DEVICE
        and RVQ_AE_MAX_SECONDS."""
        defaults = cls()
        env = {
            key.removeprefix(ENV_PREFIX).lower(): value
            for key, value in os.environ.items()
            if key.startswith(ENV_PREFIX)
        }
        return cls(
            model=env.get("model", defaults.model),
            variant=env.get("variant", defaults.variant) or None,
            subfolder=env.get("subfolder") or None,
            dav=env.get("dav", defaults.dav),
            revision=env.get("revision") or None,
            device=env.get("device", defaults.device),
            max_seconds=float(env.get("max_seconds", defaults.max_seconds)),
        )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_upload(upload: UploadFile, data: bytes) -> tuple[Tensor, int] | Tensor:
    """Waveform [channels, samples] with its rate, or latents [latent_frames, 128] from a safetensors file."""
    name = upload.filename or ""
    if name.endswith(".safetensors"):
        tensors = load_bytes(data)
        if "latents" not in tensors:
            raise HTTPException(422, "safetensors upload must contain a 'latents' tensor")
        latents = tensors["latents"]
        if latents.ndim != 2:
            raise HTTPException(422, "latents must be [latent_frames, channels]")
        return latents
    try:
        return load_audio(data)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(422, f"could not decode audio: {error}") from None


def agreement(reference: Tensor, predicted: Tensor, candidates: Tensor | None) -> EvalResponse:
    """Per codebook top 1 (and top 5 when candidates [frames, codebooks, k] are given)."""
    frames = min(reference.shape[0], predicted.shape[0])
    reference = reference[:frames]
    predicted = predicted[:frames]
    mask = reference != IGNORE
    total = mask.sum(dim=0).clamp(min=1).double()
    top1 = ((reference == predicted) & mask).sum(dim=0).double() / total
    response = EvalResponse(
        frames=frames,
        top1=top1.tolist(),
        semantic_top1=top1[0].item(),
        acoustic_top1=(
            ((reference == predicted) & mask)[:, 1:].sum() / mask[:, 1:].sum().clamp(min=1)
        ).item(),
    )
    if candidates is not None:
        hits = ((candidates[:frames] == reference.unsqueeze(-1)).any(dim=-1)) & mask
        top5 = hits.sum(dim=0).double() / total
        response.top5 = top5.tolist()
        response.semantic_top5 = top5[0].item()
        response.acoustic_top5 = (hits[:, 1:].sum() / mask[:, 1:].sum().clamp(min=1)).item()
    return response


def codes_tensor(rows: list[list[int]], books: int, name: str) -> Tensor:
    tensor = torch.tensor(rows, dtype=torch.long) if rows else torch.zeros(0, books, dtype=torch.long)
    if tensor.ndim != 2 or tensor.shape[1] != books:
        raise HTTPException(422, f"{name} must be [frames, {books}]")
    return tensor


def create_app(settings: Settings | None = None, *, encoder: CodeEncoder | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.encoder = encoder or CodeEncoder.load(
            config.model,
            variant=config.variant,
            subfolder=config.subfolder,
            dav=config.dav,
            device=config.resolved_device(),
            revision=config.revision,
        )
        app.state.lock = threading.Lock()
        yield

    app = FastAPI(title="RVQ AE MiniMax M3", version=__version__, lifespan=lifespan)

    def codec(request: Request) -> CodeEncoder:
        encoder_: CodeEncoder = request.app.state.encoder
        return encoder_

    def run(request: Request, upload: UploadFile, data: bytes, topk: int) -> EncodeResult:
        codec_ = codec(request)
        source = load_upload(upload, data)
        with request.app.state.lock:
            if isinstance(source, tuple):
                audio, rate = source
                if audio.shape[-1] / rate > config.max_seconds:
                    raise HTTPException(413, f"audio longer than {config.max_seconds} seconds")
                return codec_.encode(audio, rate, topk=topk)
            if source.shape[0] / codec_.dav.sample_rate * codec_.dav.hop > config.max_seconds:
                raise HTTPException(413, f"latents cover more than {config.max_seconds} seconds")
            return codec_.encode_latents(source, topk=topk)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/model", response_model=ModelInfo)
    async def model(request: Request) -> ModelInfo:
        codec_ = codec(request)
        return ModelInfo(
            model=config.model,
            variant=config.variant,
            dav=config.dav,
            device=str(codec_.device),
            parameters=codec_.model.parameter_count(),
            frame_rate=FRAME_RATE,
            sample_rate=codec_.dav.sample_rate,
            window_frames=codec_.window,
            codebook_vocab_sizes=list(codec_.model.cfg.codebook_vocab_sizes),
            max_seconds=config.max_seconds,
        )

    @app.post("/encode", response_model=EncodeResponse)
    async def encode(
        request: Request,
        file: Annotated[UploadFile, File()],
        topk: Annotated[int, Form(ge=0, le=50)] = 0,
        confidence: Annotated[bool, Form()] = True,
        output: Annotated[Literal["json", "safetensors"], Form()] = "json",
    ) -> Response | EncodeResponse:
        data = await file.read()
        result = await run_in_threadpool(run, request, file, data, topk)
        if output == "safetensors":
            tensors = {"codes": result.codes, "confidence": result.confidence}
            if result.candidates is not None:
                tensors["candidates"] = result.candidates
            return Response(save_bytes(tensors), media_type="application/octet-stream")
        payload = result.to_dict()
        if not confidence:
            payload["confidence"] = None
        return EncodeResponse(**payload)  # type: ignore[arg-type]

    @app.post("/evaluate/codes", response_model=EvalResponse)
    async def evaluate_codes(request: Request, body: CodesBody) -> EvalResponse:
        books = codec(request).model.cfg.num_codebooks
        reference = codes_tensor(body.reference, books, "reference")
        predicted = codes_tensor(body.predicted, books, "predicted")
        return agreement(reference, predicted, None)

    @app.post("/evaluate/audio", response_model=EvalResponse)
    async def evaluate_audio(
        request: Request,
        file: Annotated[UploadFile, File()],
        codes: Annotated[str, Form(description="JSON [frames, codebooks] reference codes")],
    ) -> EvalResponse:
        books = codec(request).model.cfg.num_codebooks
        try:
            rows = json.loads(codes)
        except json.JSONDecodeError as error:
            raise HTTPException(422, f"codes is not valid JSON: {error}") from None
        reference = codes_tensor(rows, books, "codes")
        data = await file.read()
        result = await run_in_threadpool(run, request, file, data, 5)
        return agreement(reference, result.codes, result.candidates)

    return app


def stream_to_wav_bytes(audio: Tensor, rate: int) -> bytes:
    """Helper for clients and tests: encode [channels, samples] as WAV bytes."""

    buffer = io.BytesIO()
    soundfile.write(buffer, audio.numpy(force=True).T, rate, format="WAV")
    return buffer.getvalue()
