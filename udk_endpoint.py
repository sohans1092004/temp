"""
Standalone audio -> detection endpoint. NOT wired into api.py (no journeys,
no DB, no audio store, no downstream delivery) -- it just accepts one audio
file and returns what the current best model decides for it.

Model: the evaluate_prepped_data.py pipeline (KWS + semantic + BEATs,
per-sentence matching) run TWICE on the same speech segments -- once with
Whisper small.en, once with Parakeet TDT 0.6B v2 -- and the clip alerts if
EITHER path alerts (the stronger tier wins). Recall-first: the two models fail
differently (Whisper on clipped audio, Parakeet on short overlapped phrases).
Measured on prepped_data x 6 audio conditions (2026-09-24): 82/84 danger
clips vs 78/84 Whisper-only, same false alarms (49/96).

    uvicorn udk_endpoint:app --port 8001
    curl --data-binary @clip.mp3 http://localhost:8001/v1/detect

Body = the raw audio file (wav/mp3/m4a/ogg/aac/...; decoded by PyAV, which
faster-whisper already ships). Env:
    UDK_ENDPOINT_TOKEN   if set, requests must send X-Service-Token: <it>
    UDK_ENDPOINT_MAX_MB  upload cap, default 25
    UDK_PARAKEET_DIR     where the Parakeet ONNX model lives / downloads to
                         (default ./parakeet_model, ~2.4 GB on first start)
    PREPPED_STT_MODEL / PREPPED_DEVICE   same meaning as evaluate_prepped_data.py
"""

from __future__ import annotations

import hmac
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

from evaluate_full_system import default_device, run_file_either

MAX_BYTES = int(float(os.environ.get("UDK_ENDPOINT_MAX_MB", "25")) * 1024 * 1024)
TOKEN = os.environ.get("UDK_ENDPOINT_TOKEN")
MODELS: dict = {}
# ponytail: one global lock, requests run one at a time (the models aren't
# thread-safe). A worker pool with one model set per worker if throughput matters.
_lock = threading.Lock()
RANK = {"NONE": 0, "TRIGGER_VERIFY": 1, "TRIGGER_ALL": 2}


def load_models() -> dict:
    from beats_distress_detector import BEATsDistressDetector
    from kws import Wav2Vec2DTWSpotter
    from semantic import SentenceTransformerSemanticMatcher
    from stt import FasterWhisperSTT, ParakeetSTT
    from vad import VAD

    device = os.environ.get("PREPPED_DEVICE", default_device())
    kws = Wav2Vec2DTWSpotter(device=device)
    kws.load_references(Path(__file__).parent / "kws_references.npz")
    return {
        "vad": VAD(),
        "stt": {
            "whisper": FasterWhisperSTT(
                model_size=os.environ.get("PREPPED_STT_MODEL", "small.en"),
                device=device, compute_type="float16" if device == "cuda" else "int8",
            ),
            "parakeet": ParakeetSTT(device=device),
        },
        "kws": kws,
        "semantic": SentenceTransformerSemanticMatcher(device=device),
        "beats": BEATsDistressDetector(device=device),
    }


def decode(data: bytes) -> bytes:
    """Any container/codec -> 16 kHz mono int16 PCM, the pipeline's input format."""
    from faster_whisper import decode_audio

    audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


def _path_result(events, rows, latency) -> dict:
    alert = "TRIGGER_ALL" if any(e.decision == "TRIGGER_ALL" for e in events) else (
        "TRIGGER_VERIFY" if events else "NONE")
    return {
        "alert": alert,
        "confidence": round(max((e.confidence for e in events), default=0.0), 3),
        "udks": sorted({e.udk.udk_id for e in events if e.udk}),
        "events": [
            {"decision": e.decision, "udk_id": e.udk.udk_id if e.udk else None,
             "udk_phrase": e.udk.phrase if e.udk else None, "confidence": round(e.confidence, 3),
             "layer": e.layer, "transcript": e.transcript}
            for e in events
        ],
        "segments": [{**r, "confidence": round(r["confidence"], 3)} for r in rows],
        "processing_seconds": round(latency, 2),
    }


def detect(pcm: bytes, models: dict) -> dict:
    with _lock:
        # ponytail: the two STT paths run one after the other; run them in parallel
        # threads if the ~1 s the second path adds ever matters.
        runs = run_file_either(pcm, models["vad"], models["stt"], models["kws"], models["semantic"], models["beats"])
    paths = {name: _path_result(*r) for name, r in runs.items()}
    best = max(paths.values(), key=lambda p: (RANK[p["alert"]], p["confidence"]))
    return {
        "alert": best["alert"],  # alert if EITHER STT path alerts; the stronger tier wins
        "confidence": best["confidence"],
        "udks": sorted({u for p in paths.values() for u in p["udks"]}),
        "alerted_by": [name for name, p in paths.items() if p["alert"] != "NONE"],
        "paths": paths,
        "audio_seconds": round(len(pcm) / 32000, 2),
        "processing_seconds": round(sum(p["processing_seconds"] for p in paths.values()), 2),
    }


@asynccontextmanager
async def lifespan(_app):
    MODELS.update(load_models())  # once at startup, not per request
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": bool(MODELS)}


@app.post("/v1/detect")
async def detect_endpoint(request: Request, x_service_token: str | None = Header(None)):
    if TOKEN and not hmac.compare_digest(x_service_token or "", TOKEN):
        raise HTTPException(401, "bad or missing X-Service-Token")
    data = bytearray()
    async for chunk in request.stream():  # stream, so an oversized upload is cut off early
        data += chunk
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"audio larger than {MAX_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty body: send the audio file as the request body")
    try:
        pcm = decode(bytes(data))
    except Exception:
        raise HTTPException(415, "could not decode audio")
    if not pcm:
        raise HTTPException(422, "audio decoded to zero samples")
    # Plain def work in a threadpool so the event loop isn't blocked for the whole inference.
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(detect, pcm, MODELS)
