"""
M7 checks: the Section 11 metrics/SLO set, the journey-eviction fix for
an in-memory-growth leak, an accelerated soak-shaped run, and a
concurrency bump past M5's load test.

Acceptance criterion (Section 14): SLOs hold at target scale over a
multi-hour soak. This can't literally run for hours in a test suite --
the soak test below is accelerated (many journeys, back-to-back, no real
wall-clock delay) and checks the two things a real soak actually watches
for: unbounded memory growth (via JourneyStore's tracked entry count) and
SLO drift (via the recorded latency/delivery metrics), not calendar time
itself. See README.md for what a real multi-hour soak against this would
still need (real production traffic patterns, real infra).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

import api
from audit import AuditLog
from db import JourneyDB
from event_delivery import EventDeliveryQueue
from stt import MockSTT
from storage import AudioStore
from vad import SpeechSegment

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        print(f"[FAIL] {name} {detail}")


class SingleFixedSegmentVAD:
    def __init__(self, close_at_bytes: int):
        self.close_at_bytes = close_at_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        if len(pcm) < self.close_at_bytes:
            return []
        return [SpeechSegment(start_ms=0, end_ms=self.close_at_bytes, pcm=pcm[: self.close_at_bytes])]

    def total_ms(self, pcm: bytes) -> int:
        return len(pcm)


tmp_root = Path(tempfile.mkdtemp(prefix="udk_m7_test_"))
api._audio_store = AudioStore(tmp_root / "audio")
api._event_queue = EventDeliveryQueue(tmp_root / "events")
api._db = JourneyDB(tmp_root / "journeys.sqlite3")
api._audit_log = AuditLog(tmp_root / "audit.sqlite3")
api._store = api.JourneyStore(db=api._db)
api._downstream_deliver = lambda event: True
api._rate_limiter = api.RateLimiter(max_requests=10_000, window_s=60)
api._metrics = api.Metrics()

client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})

print("=== Metrics reflect real activity, not just request/response health ===")

resp = client.post("/v1/journeys/start", json={"user_id": "U_metrics", "personal_udk_phrase": "penguin waters the orchid quietly"})
jid = resp.json()["journey_id"]
journey = api._store.get(jid)
journey.vad = SingleFixedSegmentVAD(close_at_bytes=16)
pcm = b"P" * 16
api._stt_backend.register(pcm, "penguin waters the orchid quietly")
api._ingest_frame(journey, 0, pcm, api._audio_store)
api._ingest_frame(journey, 1, b"\x00" * 4, api._audio_store)
client.post(f"/v1/journeys/{jid}/stop", json={})

snap = client.get("/metrics").json()
check("journeys_started counted", snap["journeys"]["started"] >= 1, str(snap["journeys"]))
check("journeys_completed counted", snap["journeys"]["completed"] >= 1, str(snap["journeys"]))
check("chunks_received counted", snap["audio"]["chunks_received"] >= 2, str(snap["audio"]))
check("detection recorded by decision type", sum(snap["detections_by_decision"].values()) >= 1, str(snap["detections_by_decision"]))
check("detection latency p50 is a real number", isinstance(snap["detection_latency_s"]["p50"], float))
check("event delivery success rate is 1.0 (no failures yet)", snap["event_delivery_success_rate"] == 1.0)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Journey eviction fixes the in-memory-growth leak ===")

resp2 = client.post("/v1/journeys/start", json={"user_id": "U_evict"})
jid2 = resp2.json()["journey_id"]
client.post(f"/v1/journeys/{jid2}/stop", json={})
journey2 = api._store.get(jid2)
journey2.last_heartbeat_at = time.time() - 1000  # pretend it finished a long time ago

check("finished journey is still in memory before eviction", jid2 in api._store._journeys)
evicted = api._store.evict_finished(older_than_s=500)
check("evict_finished removes it", jid2 in evicted and jid2 not in api._store._journeys)

resp_status_after_evict = client.get(f"/v1/journeys/{jid2}")
check("GET /journeys still works after eviction (via the DB, Section 9)", resp_status_after_evict.status_code == 200 and resp_status_after_evict.json()["status"] == "COMPLETE")

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Accelerated soak-shaped run: many journeys, memory bounded, SLOs hold ===")

N_SOAK = 200
for i in range(N_SOAK):
    r = client.post("/v1/journeys/start", json={"user_id": f"U_soak_{i}", "personal_udk_phrase": f"soak phrase number {i} unique"})
    jid_i = r.json()["journey_id"]
    j = api._store.get(jid_i)
    j.vad = SingleFixedSegmentVAD(close_at_bytes=16)
    stt = MockSTT()
    j.stt = stt
    p = f"S{i}".encode().ljust(16, b"S")
    stt.register(p, f"soak phrase number {i} unique")
    api._ingest_frame(j, 0, p, api._audio_store)
    api._ingest_frame(j, 1, b"\x00" * 4, api._audio_store)
    client.post(f"/v1/journeys/{jid_i}/stop", json={})
    if i % 20 == 0:
        api._store.evict_finished(older_than_s=0)  # simulate periodic housekeeping

api._store.evict_finished(older_than_s=0)

check(
    f"in-memory journey count stays bounded after {N_SOAK} journeys (eviction keeps it small)",
    len(api._store._journeys) < 20,
    f"{len(api._store._journeys)} still resident",
)

soak_snapshot = client.get("/metrics").json()
check(f"all {N_SOAK}+ journeys detected their phrase", sum(soak_snapshot["detections_by_decision"].values()) >= N_SOAK)
check("event delivery success rate held at 100% across the soak run", soak_snapshot["event_delivery_success_rate"] == 1.0, str(soak_snapshot["event_delivery_success_rate"]))
check(
    "detection latency SLO holds across the soak run (p50 <= 1.5s)",
    soak_snapshot["detection_latency_s"]["p50"] is not None and soak_snapshot["detection_latency_s"]["p50"] <= 1.5,
    str(soak_snapshot["detection_latency_s"]),
)
check(
    "detection latency SLO holds across the soak run (p95 <= 4s)",
    soak_snapshot["detection_latency_s"]["p95"] is not None and soak_snapshot["detection_latency_s"]["p95"] <= 4.0,
    str(soak_snapshot["detection_latency_s"]),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Concurrency bump: 60 concurrent journeys (3x M5's load test) ===")

import threading

N = 60
results: list[dict] = []
errors: list[str] = []
lock = threading.Lock()


def run_journey(i: int) -> None:
    try:
        c = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})
        phrase = f"falcon reads a map upside down {i}"
        r = c.post("/v1/journeys/start", json={"user_id": f"U_conc_{i}", "personal_udk_phrase": phrase}).json()
        jid_i, token_i = r["journey_id"], r["session_token"]
        j = api._store.get(jid_i)
        j.vad = SingleFixedSegmentVAD(close_at_bytes=16)
        stt = MockSTT()
        j.stt = stt
        p = f"F{i}".encode().ljust(16, b"F")
        stt.register(p, phrase)
        with c.websocket_connect(f"/v1/journeys/{jid_i}/stream?session_token={token_i}") as ws:
            ws.send_bytes(api.FRAME_HEADER.pack(0, time.time()) + p)
            ws.receive_json()
            ws.send_bytes(api.FRAME_HEADER.pack(1, time.time()) + b"\x00" * 4)
            ws.receive_json()
            event_msg = ws.receive_json()
        with lock:
            results.append({"journey_id": jid_i, "phrase": phrase, "event": event_msg})
    except Exception as exc:  # noqa: BLE001
        with lock:
            errors.append(f"{i}: {exc!r}")


threads = [threading.Thread(target=run_journey, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check(f"all {N} concurrent journeys completed without exceptions", not errors, str(errors))
check(f"all {N} journeys produced exactly one correct event", len(results) == N and all(r["event"]["transcript"] == r["phrase"] for r in results), f"{len(results)}/{N}")

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
