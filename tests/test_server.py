import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load as load_bytes
from safetensors.torch import save as save_bytes

from conftest import tiny_config, tiny_dav
from rvq_ae.alignment import nominal_bounds
from rvq_ae.constants import IGNORE
from rvq_ae.inference import CodeEncoder
from rvq_ae.model import RvqEncoder
from rvq_ae.server.app import Settings, agreement, create_app, stream_to_wav_bytes


@pytest.fixture
def client() -> TestClient:
    codec = CodeEncoder(tiny_dav(), RvqEncoder(tiny_config(latent_channels=8)), device=torch.device("cpu"))
    app = create_app(Settings(device="cpu", max_seconds=2.0), encoder=codec)
    with TestClient(app) as client:
        yield client


def wav(seconds: float = 1.0, rate: int = 44_100) -> bytes:
    return stream_to_wav_bytes(torch.randn(2, int(seconds * rate)) * 0.1, rate)


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["codebook_vocab_sizes"] == [17, 5, 5, 5, 5, 5, 5, 5]
    assert info["frame_rate"] == 25
    assert info["window_frames"] == 8
    assert info["device"] == "cpu"


def test_encode_audio_returns_codes_per_frame(client: TestClient) -> None:
    response = client.post("/encode", files={"file": ("clip.wav", wav(), "audio/wav")}, data={"topk": "3"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frames"] == 25 and body["duration"] == 1.0
    assert len(body["codes"]) == 25 and len(body["codes"][0]) == 8
    assert len(body["confidence"]) == 25
    assert len(body["semantic_topk"][0]) == 3


def test_encode_can_return_safetensors_and_skip_confidence(client: TestClient) -> None:
    response = client.post(
        "/encode", files={"file": ("clip.wav", wav(), "audio/wav")}, data={"output": "safetensors"}
    )
    tensors = load_bytes(response.content)
    assert tensors["codes"].shape == (25, 8)
    response = client.post(
        "/encode", files={"file": ("clip.wav", wav(), "audio/wav")}, data={"confidence": "false"}
    )
    assert response.json()["confidence"] is None


def test_encode_accepts_latents(client: TestClient) -> None:
    latents = torch.randn(nominal_bounds(20)[-1], 8)
    payload = save_bytes({"latents": latents})
    response = client.post(
        "/encode", files={"file": ("latents.safetensors", payload, "application/octet-stream")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["frames"] == 20
    response = client.post(
        "/encode",
        files={"file": ("x.safetensors", save_bytes({"codes": latents}), "application/octet-stream")},
    )
    assert response.status_code == 422


def test_long_audio_is_rejected(client: TestClient) -> None:
    response = client.post("/encode", files={"file": ("clip.wav", wav(seconds=3.0), "audio/wav")})
    assert response.status_code == 413


def test_undecodable_audio_is_rejected(client: TestClient) -> None:
    response = client.post("/encode", files={"file": ("clip.wav", b"not audio", "audio/wav")})
    assert response.status_code == 422


def test_evaluate_codes(client: TestClient) -> None:
    reference = [[1, 2, 3, 4, 0, 1, 2, 3], [IGNORE] * 8, [5, 1, 1, 1, 1, 1, 1, 1]]
    predicted = [[1, 2, 3, 4, 0, 1, 2, 0], [0] * 8, [5, 1, 1, 1, 1, 1, 1, 0]]
    response = client.post("/evaluate/codes", json={"reference": reference, "predicted": predicted})
    body = response.json()
    assert body["frames"] == 3
    assert body["semantic_top1"] == 1.0
    assert body["top1"][7] == 0.0
    assert body["acoustic_top1"] == pytest.approx(12 / 14)
    assert body["top5"] is None
    response = client.post("/evaluate/codes", json={"reference": [[1, 2]], "predicted": predicted})
    assert response.status_code == 422


def test_evaluate_audio_reports_top1_and_top5(client: TestClient) -> None:
    audio = torch.randn(2, 44_100) * 0.1
    encoded = client.post(
        "/encode", files={"file": ("clip.wav", stream_to_wav_bytes(audio, 44_100), "audio/wav")}
    ).json()
    response = client.post(
        "/evaluate/audio",
        files={"file": ("clip.wav", stream_to_wav_bytes(audio, 44_100), "audio/wav")},
        data={"codes": json.dumps(encoded["codes"])},
    )
    body = response.json()
    assert body["frames"] == 25
    assert body["semantic_top1"] == 1.0 and body["acoustic_top1"] == 1.0
    assert body["semantic_top5"] == 1.0 and body["acoustic_top5"] == 1.0
    response = client.post(
        "/evaluate/audio",
        files={"file": ("clip.wav", stream_to_wav_bytes(audio, 44_100), "audio/wav")},
        data={"codes": "not json"},
    )
    assert response.status_code == 422


def test_agreement_top5_uses_candidates() -> None:
    reference = torch.tensor([[3, 1], [2, 0]])
    predicted = torch.tensor([[3, 0], [1, 0]])
    candidates = torch.tensor([[[3, 0], [0, 1]], [[1, 2], [0, 4]]])
    result = agreement(reference, predicted, candidates)
    assert result.top1 == [0.5, 0.5]
    assert result.top5 == [1.0, 1.0]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RVQ_AE_VARIANT", "v1")
    monkeypatch.setenv("RVQ_AE_DEVICE", "cpu")
    monkeypatch.setenv("RVQ_AE_MAX_SECONDS", "12")
    settings = Settings.from_env()
    assert settings.variant == "v1" and settings.device == "cpu" and settings.max_seconds == 12.0
    assert settings.resolved_device() == "cpu"
