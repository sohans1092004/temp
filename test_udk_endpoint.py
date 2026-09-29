"""Fast check of udk_endpoint.py with the models stubbed out: python test_udk_endpoint.py"""
import io
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import evaluate_full_system
import udk_endpoint as ep

seen = {"samples": None, "vad_calls": 0, "segments_by_path": {}}
WHISPER, PARAKEET = object(), object()
# which STT path "hears" the UDK in the current request; set per test below
hears = {"whisper": False, "parakeet": True}


class FakeVAD:
    def segment_speech(self, pcm):
        seen["vad_calls"] += 1
        return [SimpleNamespace(pcm=pcm, start_ms=0, end_ms=1000)]


def fake_run_file(pcm, vad, stt, *models):
    name = "whisper" if stt is WHISPER else "parakeet"
    seen["samples"] = len(pcm) // 2
    seen["segments_by_path"][name] = [id(s) for s in vad.segment_speech(pcm)]
    if not hears[name]:
        return [], [{"transcript": "hello", "decision": "NO_ACTION", "udk_id": None, "layer": "none",
                     "confidence": 0.0, "scream_score": None, "stt_suspect": False}], 0.1
    udk = SimpleNamespace(udk_id="UDK_03", phrase="Get away from me")
    ev = SimpleNamespace(decision="TRIGGER_VERIFY", udk=udk, confidence=0.71, layer="fuzzy", transcript="get away")
    row = {"transcript": "get away", "decision": "TRIGGER_VERIFY", "udk_id": "UDK_03", "layer": "fuzzy",
           "confidence": 0.71, "scream_score": None, "stt_suspect": False}
    return [ev], [row], 0.1


ep.load_models = lambda: {"vad": FakeVAD(), "stt": {"whisper": WHISPER, "parakeet": PARAKEET},
                          "kws": SimpleNamespace(spot=lambda p: None), "semantic": None,
                          "beats": SimpleNamespace(score=lambda p: 0.0)}
evaluate_full_system.run_file = fake_run_file  # what run_file_either calls, once per STT path


def wav(seconds, sr=44100):  # non-16k on purpose: endpoint must resample
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype="float32"), sr, format="WAV")
    return buf.getvalue()


with TestClient(ep.app) as c:
    assert c.get("/health").json() == {"ready": True}

    # Whisper misses, Parakeet hears it -> still an alert (the whole point of the OR)
    r = c.post("/v1/detect", content=wav(2.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alert"] == "TRIGGER_VERIFY" and body["udks"] == ["UDK_03"], body
    assert body["alerted_by"] == ["parakeet"], body
    assert body["paths"]["whisper"]["alert"] == "NONE" and body["paths"]["parakeet"]["alert"] == "TRIGGER_VERIFY"
    assert abs(seen["samples"] - 32000) < 100, seen  # 2 s at 16 kHz
    # both paths got the SAME segments, and VAD ran once per request, not once per path
    assert seen["segments_by_path"]["whisper"] == seen["segments_by_path"]["parakeet"], seen
    assert seen["vad_calls"] == 1, seen

    # neither path hears anything -> no alert
    hears["parakeet"] = False
    body = c.post("/v1/detect", content=wav(1.0)).json()
    assert body["alert"] == "NONE" and body["alerted_by"] == [] and body["udks"] == [], body
    hears["parakeet"] = True

    assert c.post("/v1/detect", content=b"").status_code == 400
    assert c.post("/v1/detect", content=b"not audio at all" * 100).status_code == 415
    ep.MAX_BYTES = 1000
    assert c.post("/v1/detect", content=wav(1.0)).status_code == 413
    ep.MAX_BYTES, ep.TOKEN = 10**8, "s3cret"
    assert c.post("/v1/detect", content=wav(1.0)).status_code == 401
    assert c.post("/v1/detect", content=wav(1.0), headers={"X-Service-Token": "s3cret"}).status_code == 200
print("udk_endpoint: all checks passed")
