"""
M3 checks: REST journey lifecycle + idempotency, WebSocket auth, sequence
buffering (in-order and out-of-order), the tail-exclusion live-detection
rule, heartbeats, and the server-side timeout (Section 3).

Uses fastapi.testclient.TestClient (in-process ASGI, no real network/port
needed) and a couple of test doubles so the protocol logic is exercised
without depending on real speech audio or a real STT model -- that
combination (real webrtcvad + real FasterWhisperSTT through this same API)
was smoke-tested manually this session; see README.md for the latency
number, since a rigorous SLO validation needs Section 12's real load-test
corpus, not a unit-level check.
"""

from __future__ import annotations

import struct
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
from udks import SAMPLE_PERSONAL_UDK
from vad import SpeechSegment

# Fresh temp-backed stores for this run -- api._db/_audio_store/_event_queue
# default to fixed OS-tempdir paths that would otherwise let a journey
# recovered from a *previous* run's leftover state (e.g. J_timeout already
# TIMED_OUT) mask what a test is actually supposed to prove.
_tmp_root = Path(tempfile.mkdtemp(prefix="udk_m3_test_"))
api._audio_store = AudioStore(_tmp_root / "audio")
api._event_queue = EventDeliveryQueue(_tmp_root / "events")
api._db = JourneyDB(_tmp_root / "journeys.sqlite3")
api._store = api.JourneyStore(db=api._db)

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


class FixedBlockVAD:
    """Test double: splits the buffer into fixed-size blocks instead of
    running real webrtcvad, so WS protocol tests don't depend on real
    speech-shaped audio -- the tail-exclusion logic in
    _run_incremental_detection is what's under test here, not VAD itself
    (that's covered by test_pipeline.py and the manual FasterWhisperSTT
    verification)."""

    def __init__(self, block_bytes: int):
        self.block_bytes = block_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        n_blocks = len(pcm) // self.block_bytes
        return [
            SpeechSegment(start_ms=i * self.block_bytes, end_ms=(i + 1) * self.block_bytes, pcm=pcm[i * self.block_bytes : (i + 1) * self.block_bytes])
            for i in range(n_blocks)
        ]

    def total_ms(self, pcm: bytes) -> int:
        return (len(pcm) // self.block_bytes) * self.block_bytes


class SingleFixedSegmentVAD:
    """Models one utterance whose segment stops growing at close_at_bytes
    while more (post-utterance) audio keeps arriving after it -- exactly
    the case the old "exclude only the list's last segment" rule got
    wrong, since a lone segment is always last."""

    def __init__(self, close_at_bytes: int):
        self.close_at_bytes = close_at_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        if len(pcm) < self.close_at_bytes:
            return []
        return [SpeechSegment(start_ms=0, end_ms=self.close_at_bytes, pcm=pcm[: self.close_at_bytes])]

    def total_ms(self, pcm: bytes) -> int:
        return len(pcm)


def frame(seq: int, pcm: bytes) -> bytes:
    return api.FRAME_HEADER.pack(seq, time.time()) + pcm


client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})

print("=== REST lifecycle + idempotency ===")

resp = client.post("/v1/journeys/start", json={"user_id": "U1", "personal_udk_phrase": "purple elephants dance at midnight"})
check("start returns 200", resp.status_code == 200, resp.text)
start_body = resp.json()
journey_id = start_body["journey_id"]
session_token = start_body["session_token"]
check("start returns session_token + stream_endpoint", bool(session_token) and start_body["stream_endpoint"].endswith("/stream"))

resp2 = client.post("/v1/journeys/start", json={"user_id": "U1", "journey_id": journey_id, "personal_udk_phrase": "purple elephants dance at midnight"})
check("retried start with same journey_id is idempotent", resp2.json()["session_token"] == session_token)

resp_status = client.get(f"/v1/journeys/{journey_id}")
check("GET journey status -> ACTIVE", resp_status.json()["status"] == "ACTIVE")

resp_bad_enroll = client.post("/v1/journeys/start", json={"user_id": "U2", "personal_udk_phrase": "hi"})
check("start with an invalid personal UDK phrase is rejected", resp_bad_enroll.status_code == 400)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== WebSocket auth ===")

rejected = False
try:
    with client.websocket_connect(f"/v1/journeys/{journey_id}/stream?session_token=WRONG_TOKEN"):
        pass
except Exception:
    rejected = True
check("wrong session_token is rejected at connect", rejected)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== WS protocol: sequencing, acks, tail-exclusion live detection, heartbeat ===")

journey = api._store.get(journey_id)
BLOCK = 8
journey.vad = FixedBlockVAD(BLOCK)
block_a = b"A" * BLOCK
block_b = b"B" * BLOCK
block_c = b"C" * BLOCK
api._stt_backend.register(block_b, "purple elephants dance at midnight")

with client.websocket_connect(f"/v1/journeys/{journey_id}/stream?session_token={session_token}") as ws:
    ws.send_bytes(frame(0, block_a))
    ack0 = ws.receive_json()
    check("frame 0 acked, no event yet (tail excluded)", ack0 == {"type": "ack", "ack_seq": 0, "server_ts": ack0["server_ts"]})

    ws.send_bytes(frame(1, block_b))
    ack1 = ws.receive_json()
    check("frame 1 acked, still no event (block A had no transcript)", ack1["ack_seq"] == 1)

    ws.send_bytes(frame(2, block_c))
    ack2 = ws.receive_json()
    check("frame 2 acked", ack2["ack_seq"] == 2)
    event_msg = ws.receive_json()
    check(
        "detection event for block B arrives once it's no longer the tail",
        event_msg["type"] == "event" and event_msg["decision"] != "NO_ACTION" and event_msg["udk_type"] == "PERSONAL",
        str(event_msg),
    )

    ws.send_text('{"type": "heartbeat"}')
    hb = ws.receive_json()
    check("heartbeat is answered", hb["type"] == "heartbeat")

events = client.get(f"/v1/journeys/{journey_id}/events").json()["events"]
check("emitted event is visible via GET /events", len(events) == 1 and events[0]["udk_type"] == "PERSONAL", str(events))

stop_resp = client.post(f"/v1/journeys/{journey_id}/stop", json={})
check("stop returns FINALIZING", stop_resp.json()["status"] == "FINALIZING")
check("journey is COMPLETE after stop", client.get(f"/v1/journeys/{journey_id}").json()["status"] == "COMPLETE")

stop_resp2 = client.post(f"/v1/journeys/{journey_id}/stop", json={})
check("stop is idempotent (no re-processing, no error)", stop_resp2.status_code == 200)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Out-of-order frame buffering (Section 3) ===")

j2, _ = api._store.create_or_get("J_reorder", "U3", MockSTT(), SAMPLE_PERSONAL_UDK)
audio_store = api._audio_store
silence = b"\x00" * 32

api._ingest_frame(j2, 2, silence, audio_store)
check("out-of-order frame (seq=2) doesn't advance the buffer yet", j2.next_expected_seq == 0 and len(j2.audio_buffer) == 0)

api._ingest_frame(j2, 0, silence, audio_store)
check("filling seq=0 advances past it but not past the still-missing seq=1", j2.next_expected_seq == 1 and len(j2.audio_buffer) == 32)

api._ingest_frame(j2, 1, silence, audio_store)
check("filling the gap (seq=1) catches the buffer up through the pending seq=2", j2.next_expected_seq == 3 and len(j2.audio_buffer) == 96)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Fix: a lone/last segment finalizes once proven closed, not stuck forever ===")

oneshot_udk = enroll("U5", "purple elephants dance at midnight").udk
oneshot_stt = MockSTT()
j4, _ = api._store.create_or_get("J_oneshot", "U5", oneshot_stt, oneshot_udk)
j4.vad = SingleFixedSegmentVAD(close_at_bytes=16)
phrase_pcm = b"P" * 16
oneshot_stt.register(phrase_pcm, "purple elephants dance at midnight")

events = api._ingest_frame(j4, 0, phrase_pcm, api._audio_store)
check("a one-shot utterance (still the buffer's tail) isn't finalized yet", events == [])

events = api._ingest_frame(j4, 1, b"\x00" * 4, api._audio_store)
check(
    "once more audio proves it's done growing, it finalizes live -- no stop() needed",
    len(events) == 1 and events[0]["udk_type"] == "PERSONAL",
    str(events),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Server-side journey timeout (Section 3) ===")

j3, _ = api._store.create_or_get("J_timeout", "U4", MockSTT(), SAMPLE_PERSONAL_UDK)
j3.last_heartbeat_at = time.time() - (api.HEARTBEAT_GRACE_S + 5)
fetched = api._store.get("J_timeout")
check("journey with no heartbeat past the grace window is auto-timed-out", fetched.status == "TIMED_OUT")

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Going dark AFTER a trigger gets a tighter, more urgent timeout ===")

j4, _ = api._store.create_or_get("J_timeout_after_trigger", "U5", MockSTT(), SAMPLE_PERSONAL_UDK)
j4.had_trigger = True
# Within the ORDINARY grace window, but past the tighter post-trigger one --
# proves the two thresholds are genuinely different, not just the status label.
j4.last_heartbeat_at = time.time() - (api.POST_TRIGGER_HEARTBEAT_GRACE_S + 2)
check("still within the ordinary grace window", api.POST_TRIGGER_HEARTBEAT_GRACE_S + 2 < api.HEARTBEAT_GRACE_S)
fetched4 = api._store.get("J_timeout_after_trigger")
check("a post-trigger journey times out sooner than an ordinary one would", fetched4.status == "TIMED_OUT_AFTER_TRIGGER")

j5, _ = api._store.create_or_get("J_no_trigger_same_gap", "U6", MockSTT(), SAMPLE_PERSONAL_UDK)
j5.had_trigger = False
j5.last_heartbeat_at = time.time() - (api.POST_TRIGGER_HEARTBEAT_GRACE_S + 2)
fetched5 = api._store.get("J_no_trigger_same_gap")
check("the SAME gap does NOT time out a journey that never had a trigger", fetched5.status == "ACTIVE")

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
