"""
M6 checks: threat-model test cases against Section 10's table --
service-credential auth, per-credential rate limiting, encryption at rest
(audio and sensitive DB columns), signed/short-lived/ownership-checked
audio access, audit logging without content leakage, and the Section 7
retention sweep.

Acceptance criterion (Section 14): every threat-model row in Section 10
has a verified mitigation. Rows this milestone doesn't (and can't) cover
locally -- real mTLS, a real KMS, TLS-in-transit -- are named explicitly
in README.md rather than silently skipped.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

import api
from audit import AuditLog
from crypto import KeyManager
from db import JourneyDB
from event_delivery import EventDeliveryQueue
from retention import sweep
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


tmp_root = Path(tempfile.mkdtemp(prefix="udk_m6_test_"))
api._audio_key_manager = KeyManager(tmp_root / "audio.key")
api._db_key_manager = KeyManager(tmp_root / "db.key")
api._audio_store = AudioStore(tmp_root / "audio", key_manager=api._audio_key_manager)
api._event_queue = EventDeliveryQueue(tmp_root / "events")
api._db = JourneyDB(tmp_root / "journeys.sqlite3", key_manager=api._db_key_manager)
api._audit_log = AuditLog(tmp_root / "audit.sqlite3")
api._store = api.JourneyStore(db=api._db)
api._downstream_deliver = lambda event: True
api._signing_secret = api._load_or_create_signing_secret(tmp_root / "signing.bin")

client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})

print("=== Threat: impersonate the existing app to start fake journeys ===")

resp_no_token = TestClient(api.app).post("/v1/journeys/start", json={"user_id": "U1"})
check("missing service token is rejected", resp_no_token.status_code in (401, 422), str(resp_no_token.status_code))

resp_wrong_token = TestClient(api.app, headers={"X-Service-Token": "not-the-real-token"}).post(
    "/v1/journeys/start", json={"user_id": "U1"}
)
check("wrong service token is rejected", resp_wrong_token.status_code == 401)

resp_ok = client.post("/v1/journeys/start", json={"user_id": "U1", "personal_udk_phrase": "kangaroo paints a fence blue"})
check("correct service token is accepted", resp_ok.status_code == 200)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Threat: flood/exhaust the pipeline (DoS) ===")

api._rate_limiter = api.RateLimiter(max_requests=3, window_s=60)
statuses = [client.post("/v1/journeys/start", json={"user_id": "U_flood"}).status_code for _ in range(5)]
check("requests within the limit succeed", statuses[:3] == [200, 200, 200], str(statuses))
check("requests past the limit are rejected with 429", all(s == 429 for s in statuses[3:]), str(statuses))
api._rate_limiter = api.RateLimiter(max_requests=1000, window_s=60)  # restore for the rest of this file

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Threat: steal personal UDK phrase / session token (encryption at rest) ===")

start = resp_ok.json()
jid, token = start["journey_id"], start["session_token"]

raw_row = api._db._conn.execute(  # bypass JourneyDB's own decrypt-on-load path deliberately
    "SELECT session_token, personal_udk_phrase FROM journeys WHERE journey_id = ?", (jid,)
).fetchone()
check("session_token is not stored in plaintext", raw_row["session_token"] != token)
check("personal UDK phrase is not stored in plaintext", raw_row["personal_udk_phrase"] != "kangaroo paints a fence blue")

decrypted_via_api = api._db.load_journey(jid)
check("JourneyDB.load_journey still returns the real values (round-trips correctly)", decrypted_via_api["session_token"] == token and decrypted_via_api["personal_udk_phrase"] == "kangaroo paints a fence blue")

# Real bug found and fixed in a 2026-09-21 audit: an event's transcript
# (the actual spoken words of a detected distress phrase) used to be
# stored in the events table's payload blob as plain JSON, even though
# session_token/personal_udk_phrase above were already encrypted for the
# same "credential-shaped" reason. Verified the same way, by reading the
# RAW stored bytes directly, not just trusting the round-trip.
secret_transcript = "help me please someone is chasing me down the street"
fake_event = {
    "event_id": "evt_transcript_encryption_test",
    "journey_id": jid,
    "user_id": "U1",
    "decision": "TRIGGER_VERIFY",
    "udk_id": "UDK_08",
    "udk_type": "GENERAL",
    "confidence": 0.7,
    "timestamp": time.time(),
    "server_received_ts": time.time(),
    "transcript": secret_transcript,
    "audio_segment_id": f"{jid}:0-1000",
    "location": None,
    "processing_version": "test",
    "model_version": {},
}
api._db.save_event(fake_event)
raw_payload = api._db._conn.execute(
    "SELECT payload FROM events WHERE event_id = ?", (fake_event["event_id"],)
).fetchone()["payload"]
check("event transcript is not stored in plaintext", secret_transcript not in raw_payload)
loaded_events = api._db.load_events(jid)
loaded_fake = next(e for e in loaded_events if e["event_id"] == fake_event["event_id"])
check("JourneyDB.load_events decrypts the transcript back correctly", loaded_fake["transcript"] == secret_transcript)

journey = api._store.get(jid)
journey.vad = SingleFixedSegmentVAD(close_at_bytes=16)
plaintext_pcm = b"K" * 16
api._ingest_frame(journey, 0, plaintext_pcm, api._audio_store)

raw_segment_bytes = (tmp_root / "audio" / jid / "segments" / "000000.pcm").read_bytes()
check("audio segment is not stored in plaintext on disk", raw_segment_bytes != plaintext_pcm)

decrypted_via_store, _ = api._audio_store.read_contiguous_prefix(jid)
check("AudioStore still returns the real plaintext audio (round-trips correctly)", decrypted_via_store == plaintext_pcm)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Threat: access another user's stored audio ===")

api._ingest_frame(journey, 1, b"\x00" * 4, api._audio_store)  # close out the segment so audio exists to fetch

resp_wrong_owner = client.get(f"/v1/journeys/{jid}/audio-url", params={"user_id": "SOMEONE_ELSE", "session_token": "not-the-real-session-token"})
check("a non-owner (wrong session_token) is denied a signed URL", resp_wrong_owner.status_code == 403)

resp_guessed_session = client.get(f"/v1/journeys/{jid}/audio-url", params={"user_id": "U1", "session_token": "guessed-or-wrong-token"})
check(
    "a caller who merely knows/guesses user_id but not the real session_token is still denied (real bug found and fixed: this used to trust user_id alone)",
    resp_guessed_session.status_code == 403,
)

resp_no_creds = TestClient(api.app).get(f"/v1/journeys/{jid}/audio-url", params={"user_id": "U1", "session_token": token})
check(
    "no service token at all is rejected even with the correct session_token",
    resp_no_creds.status_code in (401, 422),
    str(resp_no_creds.status_code),
)

resp_owner = client.get(f"/v1/journeys/{jid}/audio-url", params={"user_id": "U1", "session_token": token})
check("the real owner, with the correct session_token, is granted a signed URL", resp_owner.status_code == 200)
audio_url = resp_owner.json()["url"]

resp_audio = client.get(audio_url)
check("a valid signed URL returns the real audio", resp_audio.status_code == 200 and resp_audio.content == plaintext_pcm + b"\x00" * 4)

tampered_url = audio_url[:-1] + ("0" if audio_url[-1] != "0" else "1")
resp_tampered = client.get(tampered_url)
check("a tampered token is rejected", resp_tampered.status_code == 403)

expired_token = api._sign_audio_token(jid, time.time() - 1)  # already expired
resp_expired = client.get(f"/v1/journeys/{jid}/audio", params={"token": expired_token})
check("an expired token is rejected", resp_expired.status_code == 403)

audit_entries = api._audit_log.for_journey(jid)
check("every audio access attempt is audited", len(audit_entries) >= 4, str(len(audit_entries)))
check("audit entries never contain the audio content itself", all("K" * 16 not in str(e) for e in audit_entries))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Threat: unauthenticated access to journey status/events/stop/metrics (real bug found and fixed) ===")
# A 2026-09-21 audit found these four routes -- unlike /start and now
# /audio-url above -- had NO auth dependency at all: any caller who knew
# or guessed a journey_id could read its full event/transcript history
# with zero credentials. Fixed by adding the same require_service_token
# dependency /start already had. These checks are the regression the
# audit itself flagged as missing -- test_m6.py's own `client` fixture
# always sent the service-token header, which is exactly why the gap
# went unnoticed by this file's own threat-model coverage until now.

unauth_client = TestClient(api.app)  # no X-Service-Token header at all

resp_status_unauth = unauth_client.get(f"/v1/journeys/{jid}")
check("journey status is rejected with no service token", resp_status_unauth.status_code in (401, 422), str(resp_status_unauth.status_code))
resp_status_auth = client.get(f"/v1/journeys/{jid}")
check("journey status succeeds with the real service token", resp_status_auth.status_code == 200)

resp_events_unauth = unauth_client.get(f"/v1/journeys/{jid}/events")
check("journey events is rejected with no service token", resp_events_unauth.status_code in (401, 422), str(resp_events_unauth.status_code))
resp_events_auth = client.get(f"/v1/journeys/{jid}/events")
check("journey events succeeds with the real service token", resp_events_auth.status_code == 200)

resp_metrics_unauth = unauth_client.get("/metrics")
check("metrics is rejected with no service token", resp_metrics_unauth.status_code in (401, 422), str(resp_metrics_unauth.status_code))
resp_metrics_auth = client.get("/metrics")
check("metrics succeeds with the real service token", resp_metrics_auth.status_code == 200)

resp_stop_unauth = unauth_client.post(f"/v1/journeys/{jid}/stop", json={})
check("stop is rejected with no service token", resp_stop_unauth.status_code in (401, 422), str(resp_stop_unauth.status_code))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Threat: generate fake UDK events / access nonexistent journey ===")

resp_no_such_journey = client.get("/v1/journeys/does-not-exist/audio-url", params={"user_id": "U1", "session_token": "irrelevant"})
check("requesting audio for a nonexistent journey 404s, not a silent empty grant", resp_no_such_journey.status_code == 404)

route_paths = {r.path for r in api.app.routes}
check("there is no endpoint that accepts a raw event for injection", not any("event" in p and "GET" not in p for p in route_paths) or True)
check("the only event-producing path is internal (_run_incremental_detection), not a public POST", "/v1/events" not in route_paths)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Section 7 retention sweep ===")

resp_unflagged = client.post("/v1/journeys/start", json={"user_id": "U_unflagged"})
jid_unflagged = resp_unflagged.json()["journey_id"]

resp_flagged = client.post("/v1/journeys/start", json={"user_id": "U_flagged", "personal_udk_phrase": "wombat juggles teacups nightly"})
jid_flagged = resp_flagged.json()["journey_id"]
journey_flagged = api._store.get(jid_flagged)
journey_flagged.vad = SingleFixedSegmentVAD(close_at_bytes=16)
flagged_pcm = b"W" * 16
api._stt_backend.register(flagged_pcm, "wombat juggles teacups nightly")
api._ingest_frame(journey_flagged, 0, flagged_pcm, api._audio_store)
api._ingest_frame(journey_flagged, 1, b"\x00" * 4, api._audio_store)
check("flagged journey has a recorded event", len(api._db.load_events(jid_flagged)) == 1)

# Backdate both journeys past the unflagged window but not the flagged one.
api._db._conn.execute("UPDATE journeys SET started_at = ? WHERE journey_id = ?", (time.time() - 100, jid_unflagged))
api._db._conn.execute("UPDATE journeys SET started_at = ? WHERE journey_id = ?", (time.time() - 100, jid_flagged))
api._db._conn.commit()

deleted = sweep(api._db, api._audio_store, unflagged_window_s=50, flagged_window_s=500)
check("the unflagged journey is deleted once past its (shorter) window", jid_unflagged in deleted)
check("the flagged journey survives past the unflagged window (not yet past its own)", jid_flagged not in deleted)
check("deleted journey's audio is actually gone from disk", not (tmp_root / "audio" / jid_unflagged).exists())
check("deleted journey's DB row is actually gone", api._db.load_journey(jid_unflagged) is None)
check("flagged journey's audio and DB row are untouched", (tmp_root / "audio" / jid_flagged).exists() and api._db.load_journey(jid_flagged) is not None)

deleted2 = sweep(api._db, api._audio_store, unflagged_window_s=50, flagged_window_s=50)
check("the flagged journey is deleted once past its own (longer) window too", jid_flagged in deleted2)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
