from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model: str
    variant: str | None
    dav: str
    device: str
    parameters: int
    frame_rate: int
    sample_rate: int
    window_frames: int
    codebook_vocab_sizes: list[int]
    max_seconds: float


class EncodeResponse(BaseModel):
    frames: int
    frame_rate: int
    duration: float
    latent_frames: int
    codes: list[list[int]]
    confidence: list[list[float]] | None
    semantic_topk: list[list[int]] | None


class CodesBody(BaseModel):
    reference: list[list[int]] = Field(description="[frames, codebooks] codes; -100 masks a frame")
    predicted: list[list[int]] = Field(description="[frames, codebooks] codes")


class EvalResponse(BaseModel):
    frames: int
    top1: list[float]
    semantic_top1: float
    acoustic_top1: float
    top5: list[float] | None = None
    semantic_top5: float | None = None
    acoustic_top5: float | None = None
