"""
M5 checks: durable journey/event persistence (db.py, SQLite standing in
for Postgres), the full Section 9 decision-event schema, and a modest
concurrent-journey load smoke test.

Acceptance criterion (Section 14): event delivery success rate SLO met
under load. The concurrency count below (20) is a prototype-scale smoke
test proving no cross-journey corruption and no delivery loss under
concurrent access -- not the real target ("thousands of concurrent
journeys" per the brief's Section 20), which needs real infrastructure
load-testing (Section 12), out of scope for a local prototype.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

import api
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
    """Same test double as test_m3.py/test_m4.py."""

    def __init__(self, close_at_bytes: int):
        self.close_at_bytes = close_at_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        if len(pcm) < self.close_at_bytes:
            return []
        return [SpeechSegment(start_ms=0, end_ms=self.close_at_bytes, pcm=pcm[: self.close_at_bytes])]

    def total_ms(self, pcm: bytes) -> int:
        return len(pcm)


tmp_root = Path(tempfile.mkdtemp(prefix="udk_m5_test_"))
api._audio_store = AudioStore(tmp_root / "audio")
api._event_queue = EventDeliveryQueue(tmp_root / "events")
api._db = JourneyDB(tmp_root / "journeys.sqlite3")
api._store = api.JourneyStore(db=api._db)
api._downstream_deliver = lambda event: True
# The default rate limit (Section 10, M6) is tuned for one real client's
# realistic traffic, not 20 threads sharing one dev service-token in a
# load test -- test_m6.py exercises the limit itself with a tight window.
api._rate_limiter = api.RateLimiter(max_requests=1000, window_s=60)

client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})

print("=== Full crash recovery: session_token and personal UDK now survive too (M5's fix for M4's gap) ===")

start = client.post("/v1/journeys/start", json={"user_id": "U_full_crash", "personal_udk_phrase": "tangerine kites over the harbor"}).json()
jid1, token1 = start["journey_id"], start["session_token"]

api._store._journeys.pop(jid1)  # simulate a full process restart

journey1, created1 = api._store.create_or_get(jid1, "U_full_crash", MockSTT(), api.SAMPLE_PERSONAL_UDK, api._audio_store)
check("recovery reuses the SAME session_token, not a fresh one", journey1.session_token == token1)
check("recovery reconstructs the enrolled personal UDK phrase from durable storage", journey1.personal_udk.phrase == "tangerine kites over the harbor")
check("recovered journey status matches its pre-crash status", journey1.status == "ACTIVE")

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Durable events/status survive a crash even without recovery being triggered ===")

start2 = client.post("/v1/journeys/start", json={"user_id": "U_durable", "personal_udk_phrase": "octopus reads a paperback"}).json()
jid2, token2 = start2["journey_id"], start2["session_token"]
journey2 = api._store.get(jid2)
journey2.vad = SingleFixedSegmentVAD(close_at_bytes=16)
stt2 = MockSTT()
journey2.stt = stt2
phrase_pcm2 = b"O" * 16
stt2.register(phrase_pcm2, "octopus reads a paperback")

api._ingest_frame(journey2, 0, phrase_pcm2, api._audio_store)
api._ingest_frame(journey2, 1, b"\x00" * 4, api._audio_store)
check("event detected pre-crash", len(journey2.events) == 1)

api._store._journeys.pop(jid2)  # crash: in-memory journey (and its .events list) is gone, no recovery triggered

resp_events = client.get(f"/v1/journeys/{jid2}/events")
check("GET /events still works via the DB, without needing recovery first", resp_events.status_code == 200 and len(resp_events.json()["events"]) == 1, resp_events.text)

resp_status = client.get(f"/v1/journeys/{jid2}")
check("GET /journeys status still works via the DB", resp_status.status_code == 200 and resp_status.json()["status"] == "ACTIVE", resp_status.text)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Full Section 9 decision-event schema ===")

event = resp_events.json()["events"][0]
expected_fields = {
    "event_id", "journey_id", "user_id", "decision", "udk_id", "udk_type", "confidence",
    "timestamp", "server_received_ts", "transcript", "audio_segment_id", "location",
    "processing_version", "model_version",
}
check("event has every Section 9 field", expected_fields <= event.keys(), str(sorted(event.keys())))
check("model_version is tracked per component", set(event["model_version"].keys()) == {"vad", "stt", "kws", "scream_detector"}, str(event["model_version"]))
check("timestamp (audio time) predates or equals server_received_ts", event["timestamp"] <= event["server_received_ts"])

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Concurrent-journey load smoke test ===")

N = 20
results: list[dict] = []
results_lock = threading.Lock()
errors: list[str] = []


def run_journey(i: int) -> None:
    try:
        c = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})
        phrase = f"zebra hums quietly under lamp number {i}"
        resp = c.post("/v1/journeys/start", json={"user_id": f"U_load_{i}", "personal_udk_phrase": phrase}).json()
        jid, token = resp["journey_id"], resp["session_token"]

        journey = api._store.get(jid)
        journey.vad = SingleFixedSegmentVAD(close_at_bytes=16)
        stt = MockSTT()
        journey.stt = stt
        phrase_pcm = f"P{i}".encode().ljust(16, b"P")
        stt.register(phrase_pcm, phrase)

        with c.websocket_connect(f"/v1/journeys/{jid}/stream?session_token={token}") as ws:
            ws.send_bytes(api.FRAME_HEADER.pack(0, time.time()) + phrase_pcm)
            ws.receive_json()  # ack only, still the tail
            ws.send_bytes(api.FRAME_HEADER.pack(1, time.time()) + b"\x00" * 4)
            ws.receive_json()  # ack
            event_msg = ws.receive_json()  # event

        with results_lock:
            results.append({"journey_id": jid, "phrase": phrase, "event": event_msg})
    except Exception as exc:  # noqa: BLE001 -- surfaced via the errors list, not swallowed
        with results_lock:
            errors.append(f"journey {i}: {exc!r}")


threads = [threading.Thread(target=run_journey, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check(f"all {N} concurrent journeys completed without exceptions", not errors, str(errors))
check(f"all {N} journeys produced exactly one event each", len(results) == N, f"got {len(results)}")
check(
    "no cross-journey contamination -- each event belongs to its own journey and phrase",
    all(r["event"]["journey_id"] == r["journey_id"] and r["event"]["transcript"] == r["phrase"] for r in results),
)

event_ids = {r["event"]["event_id"] for r in results}
check("all event_ids are unique across concurrent journeys", len(event_ids) == N, f"{len(event_ids)} unique of {N}")

pending = api._event_queue.pending()
check("event delivery success rate under load: 100% (queue empty)", pending == [], f"{len(pending)} still pending")

db_rows = [api._db.load_journey(r["journey_id"]) for r in results]
check("every concurrent journey's metadata landed correctly in the DB", all(row is not None for row in db_rows))

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
