"""
M4 checks: failure injection against Part 1's reliability mechanisms --
duplicate/retried frame delivery, a simulated process crash + restart
(in-memory state wiped, durable disk state survives), and the outbound
event delivery queue (enqueue -> failed delivery -> retry -> delivered).

Acceptance criterion (Section 14): no data loss across the injected
scenarios below -- audio isn't duplicated or lost, detections aren't
lost even across a "crash", and outbound events aren't dropped on a
downstream delivery failure.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

import api
from db import JourneyDB
from enrollment import enroll
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
    """Same test double as test_m3.py: one segment that stops growing at
    close_at_bytes, so detection can be checked without real speech audio."""

    def __init__(self, close_at_bytes: int):
        self.close_at_bytes = close_at_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        if len(pcm) < self.close_at_bytes:
            return []
        return [SpeechSegment(start_ms=0, end_ms=self.close_at_bytes, pcm=pcm[: self.close_at_bytes])]

    def total_ms(self, pcm: bytes) -> int:
        return len(pcm)


# Fresh temp-backed stores for this run -- api._audio_store/_event_queue/_db
# default to fixed OS-tempdir paths that would otherwise let a leftover
# journey row from a *previous* run (e.g. J_crash already recovered) mask
# what this run's crash-recovery test is actually supposed to prove.
tmp_root = Path(tempfile.mkdtemp(prefix="udk_m4_test_"))
api._audio_store = AudioStore(tmp_root / "audio")
api._event_queue = EventDeliveryQueue(tmp_root / "events")
api._db = JourneyDB(tmp_root / "journeys.sqlite3")
api._store = api.JourneyStore(db=api._db)
api._downstream_deliver = lambda event: True

print("=== Idempotent frame ingestion (duplicate/retried frames) ===")

j1, _ = api._store.create_or_get("J_dup", "U1", MockSTT(), enroll("U1", "purple elephants dance at midnight").udk)
silence = b"\x00" * 32

api._ingest_frame(j1, 0, silence, api._audio_store)
api._ingest_frame(j1, 1, silence, api._audio_store)
check("two distinct frames -> buffer has both", len(j1.audio_buffer) == 64 and j1.next_expected_seq == 2)

events = api._ingest_frame(j1, 0, silence, api._audio_store)  # retry of an already-consumed frame
check("retrying an already-consumed frame is a no-op on the buffer", len(j1.audio_buffer) == 64 and j1.next_expected_seq == 2)
check("retry produces no new events", events == [])

manifest = api._audio_store.finalize("J_dup")
check("retry doesn't duplicate the segment on disk", len(manifest.segments) == 2, str(manifest.segments))

try:
    api._audio_store.write_segment("J_dup", 0, b"\xff" * 32)  # same seq, different content
    check("same seq + different content raises (not a silent overwrite)", False)
except ValueError:
    check("same seq + different content raises (not a silent overwrite)", True)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Simulated crash + restart (in-memory state wiped, disk survives) ===")

crash_udk = enroll("U2", "purple elephants dance at midnight").udk
j2, _ = api._store.create_or_get("J_crash", "U2", MockSTT(), crash_udk)
j2.vad = SingleFixedSegmentVAD(close_at_bytes=16)
phrase_pcm = b"P" * 16
j2.stt.register(phrase_pcm, "purple elephants dance at midnight")

api._ingest_frame(j2, 0, phrase_pcm, api._audio_store)  # buffered, but not yet finalizable (still the tail)
check("pre-crash: phrase buffered but not yet detected (still growing)", j2.events == [])

# Simulate a process crash: the in-memory journey is gone, the durable
# audio on disk is not.
api._store._journeys.pop("J_crash")

recovered_stt = MockSTT()
recovered_stt.register(phrase_pcm, "purple elephants dance at midnight")
j2_recovered, created = api._store.create_or_get("J_crash", "U2", recovered_stt, crash_udk, api._audio_store)
check("recovery creates a fresh journey object (old one is gone)", created)
check("recovered buffer matches what was durably stored pre-crash", bytes(j2_recovered.audio_buffer) == phrase_pcm)
check("recovered next_expected_seq picks up where the crash left off", j2_recovered.next_expected_seq == 1)

j2_recovered.vad = SingleFixedSegmentVAD(close_at_bytes=16)
api._ingest_frame(j2_recovered, 1, b"\x00" * 4, api._audio_store)  # more audio proves the segment closed
check(
    "backlog detection resumes after recovery -- no data lost across the crash",
    len(j2_recovered.events) == 1 and j2_recovered.events[0]["udk_type"] == "PERSONAL",
    str(j2_recovered.events),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Outbound event delivery: durable queue survives a downstream failure ===")

api._downstream_deliver = lambda event: False  # simulate the safety platform being unreachable

j3_udk = enroll("U3", "purple elephants dance at midnight").udk
j3, _ = api._store.create_or_get("J_deliver", "U3", MockSTT(), j3_udk)
j3.vad = SingleFixedSegmentVAD(close_at_bytes=16)
phrase_pcm2 = b"Q" * 16
j3.stt.register(phrase_pcm2, "purple elephants dance at midnight")

api._ingest_frame(j3, 0, phrase_pcm2, api._audio_store)
api._ingest_frame(j3, 1, b"\x00" * 4, api._audio_store)
check("detection happened despite downstream being unreachable", len(j3.events) == 1, str(j3.events))

pending = api._event_queue.pending()
check("undelivered event stays durably queued, not dropped", len(pending) == 1 and pending[0]["event_id"] == j3.events[0]["event_id"])

deliveries_seen = []
api._downstream_deliver = lambda event: (deliveries_seen.append(event["event_id"]) or True)
delivered, remaining = api._event_queue.drain(api._downstream_deliver)
check("recovery: draining the queue delivers the backlog", delivered == 1 and remaining == 0)
check("no duplicate delivery of the same event", deliveries_seen == [j3.events[0]["event_id"]])
check("queue is empty after a successful drain", api._event_queue.pending() == [])

delivered2, remaining2 = api._event_queue.drain(api._downstream_deliver)
check("draining an empty queue is a safe no-op", delivered2 == 0 and remaining2 == 0)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Full network loss: WS drops mid-journey, client reconnects and resumes ===")

client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})
start = client.post("/v1/journeys/start", json={"user_id": "U4", "personal_udk_phrase": "purple elephants dance at midnight"}).json()
jid, token = start["journey_id"], start["session_token"]

with client.websocket_connect(f"/v1/journeys/{jid}/stream?session_token={token}") as ws:
    ws.send_bytes(api.FRAME_HEADER.pack(0, time.time()) + b"\x00" * 32)
    ack0 = ws.receive_json()
    check("frame 0 acked before the drop", ack0["ack_seq"] == 0)
# `with` block exit disconnects the socket -- simulates the network dying
# mid-journey (Section 3's "full network loss" row). Nothing was sent for
# seq 1 yet, so the client's own local buffer (out of scope here -- that's
# the mobile app) would still hold it to resend on reconnect.

journey_mid_drop = api._store.get(jid)
check("journey survives the drop (state isn't tied to the socket)", journey_mid_drop is not None and journey_mid_drop.status == "ACTIVE")

with client.websocket_connect(f"/v1/journeys/{jid}/stream?session_token={token}") as ws2:
    # Resuming client replays from ack_seq + 1 (Section 9) -- including a
    # harmless re-send of seq 0 in case ITS ack was what got lost, not
    # the frame itself; the idempotency fix above makes that safe.
    ws2.send_bytes(api.FRAME_HEADER.pack(0, time.time()) + b"\x00" * 32)
    ack_retry = ws2.receive_json()
    ws2.send_bytes(api.FRAME_HEADER.pack(1, time.time()) + b"\x00" * 32)
    ack1 = ws2.receive_json()
    check("reconnect resumes on the same journey, no duplicate audio", ack1["ack_seq"] == 1)

manifest = api._audio_store.finalize(jid)
check("exactly 2 segments stored across the disconnect/reconnect, not 3", len(manifest.segments) == 2, str(manifest.segments))

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
