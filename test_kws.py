"""
kws.py checks: MockKWS matching logic, the KWSMatch/threshold contract,
and UDKEngine's STT+KWS fusion (Section 4's Approach E) -- the actual
point of building this at all: proving KWS can produce a decision on its
own when STT fails, and that independent-branch agreement is treated as
real cross-validation, not just "pick the bigger number."

Fast and portable -- the real Wav2Vec2DTWSpotter backend was verified
manually against real TTS audio this session (see README.md for the
actual DTW-distance numbers); it needs torch+transformers and a model
download, so it isn't part of the routine automated suite, same call
stt.py's FasterWhisperSTT verification made back in M1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import api
from db import JourneyDB
from enrollment import enroll
from event_delivery import EventDeliveryQueue
from kws import KWSMatch, MockKWS
from storage import AudioStore
from stt import MockSTT
from udk_engine import UDKEngine
from udks import GENERAL_UDKS
from vad import SpeechSegment

_tmp_root = Path(tempfile.mkdtemp(prefix="udk_kws_test_"))
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


print("=== MockKWS ===")

kws = MockKWS()
phrase_pcm = b"\x01\x02" * 50
unrelated_pcm = b"\x03\x04" * 50

check("unregistered audio has no match", kws.spot(unrelated_pcm) is None)

kws.register(phrase_pcm, "UDK_01", distance=0.15)
match = kws.spot(phrase_pcm)
check("registered audio returns the registered match", match == KWSMatch(phrase_id="UDK_01", distance=0.15), str(match))
check("a different clip still has no match", kws.spot(unrelated_pcm) is None)

ref_pcm = b"reference clip bytes"
kws.enroll("UDK_02", ref_pcm)
check("enroll() registers the clip as a self-match (distance 0), like the real backend would give", kws.spot(ref_pcm) == KWSMatch(phrase_id="UDK_02", distance=0.0))

kws.clear("UDK_02")
check("clear() removes references for that phrase_id (Section 5: re-enrollment deactivates the old phrase)", kws.spot(ref_pcm) is None)
check("clear() doesn't disturb other phrases' registrations", kws.spot(phrase_pcm) == KWSMatch(phrase_id="UDK_01", distance=0.15))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Personal UDK enrollment also registers a KWS reference (Section 4 extended to personal UDKs) ===")

enroll_kws = MockKWS()
enroll_stt = MockSTT()
sample_pcm = b"\x05\x06" * 40
enroll_stt.register(sample_pcm, "iguana naps on the windowsill")
enroll_result = enroll("U_kws_enroll", "iguana naps on the windowsill", sample_pcm=sample_pcm, stt=enroll_stt, kws=enroll_kws)
check("enrollment still succeeds with kws given", enroll_result.ok, str(enroll_result.errors))
check(
    "the sample audio was registered as this UDK's KWS reference",
    enroll_kws.spot(sample_pcm) is not None and enroll_kws.spot(sample_pcm).phrase_id == enroll_result.udk.udk_id,
)

no_sample_result = enroll("U_kws_enroll2", "walrus paints the fence orange", kws=MockKWS())
check("enrollment without sample_pcm still succeeds (kws enrollment is best-effort, not required)", no_sample_result.ok, str(no_sample_result.errors))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== UDKEngine + KWS fusion (Section 4's Approach E) ===")

engine = UDKEngine(GENERAL_UDKS)

# UDK_01 = "Call the police"
decision = engine.decide("", now_s=0.0, kws_match=KWSMatch(phrase_id="UDK_01", distance=0.2))
check(
    "KWS alone produces a decision when STT gets nothing (the actual point: STT outage doesn't blind the system)",
    decision.udk is not None and decision.udk.udk_id == "UDK_01" and decision.decision != "NO_ACTION",
    str(decision),
)

decision2 = engine.decide("call the police right now", now_s=1.0)
check(
    "STT alone still works unaffected when no kws_match is given",
    decision2.udk is not None and decision2.udk.udk_id == "UDK_01" and decision2.decision == "TRIGGER_ALL",
    str(decision2),
)

# UDK_14 = "I don't feel safe here" (verify_by_default -- common-in-conversation)
verify_engine = UDKEngine(GENERAL_UDKS)
stt_only = verify_engine.decide("I don't feel safe here", now_s=10.0)
check("verify-by-default phrase via STT alone stays TRIGGER_VERIFY", stt_only.decision == "TRIGGER_VERIFY", str(stt_only))

corroborated_engine = UDKEngine(GENERAL_UDKS)
corroborated = corroborated_engine.decide(
    "I don't feel safe here", now_s=20.0, kws_match=KWSMatch(phrase_id="UDK_14", distance=0.15)
)
check(
    "independent STT+KWS agreement escalates a verify-by-default phrase straight to TRIGGER_ALL",
    corroborated.decision == "TRIGGER_ALL" and corroborated.reason == "independent STT+KWS agreement",
    str(corroborated),
)

# Real bug found and fixed: a borderline KWS distance (comfortably under
# spot()'s own 0.18 match threshold, but not under the stricter
# KWS_CORROBORATION_MAX_DISTANCE=0.15) used to still corroborate and
# force an unconfirmed TRIGGER_ALL. Real audio exposed this at distance
# 0.167-0.172 (README.md's "Stress-testing" section) combining with a
# borderline STT confidence (0.755, fuzzy) on an ordinary sentence.
borderline_engine = UDKEngine(GENERAL_UDKS)
borderline = borderline_engine.decide(
    "I don't feel safe here", now_s=25.0, kws_match=KWSMatch(phrase_id="UDK_14", distance=0.17)
)
check(
    "a KWS match that clears spot()'s 0.18 threshold but not the stricter 0.15 corroboration margin doesn't force TRIGGER_ALL",
    borderline.decision == "TRIGGER_VERIFY" and borderline.reason != "independent STT+KWS agreement",
    str(borderline),
)

# Disagreement: STT and KWS land on different UDKs -- take the stronger
# one, don't silently combine them into a false corroboration.
disagree_engine = UDKEngine(GENERAL_UDKS)
disagreement = disagree_engine.decide(
    "someone is trying to hurt me", now_s=30.0, kws_match=KWSMatch(phrase_id="UDK_08", distance=0.25)
)
check(
    "disagreeing branches don't falsely corroborate -- the higher-confidence one (exact STT match) wins",
    disagreement.udk is not None and disagreement.udk.udk_id == "UDK_12" and disagreement.reason != "independent STT+KWS agreement",
    str(disagreement),
)

# An unrelated KWS phrase_id this engine doesn't recognize is ignored, not guessed at.
unknown_engine = UDKEngine(GENERAL_UDKS)
unknown_kws = unknown_engine.decide("just chatting about the weather", now_s=40.0, kws_match=KWSMatch(phrase_id="NOT_A_REAL_UDK", distance=0.1))
check("an unrecognized KWS phrase_id is ignored, not treated as a match", unknown_kws.decision == "NO_ACTION", str(unknown_kws))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Through the real live API: KWS survives a total STT outage ===")


class SingleFixedSegmentVAD:
    def __init__(self, close_at_bytes: int):
        self.close_at_bytes = close_at_bytes

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        if len(pcm) < self.close_at_bytes:
            return []
        return [SpeechSegment(start_ms=0, end_ms=self.close_at_bytes, pcm=pcm[: self.close_at_bytes])]

    def total_ms(self, pcm: bytes) -> int:
        return len(pcm)


class DeadSTT:
    """Simulates a total STT outage (Section 3's row) -- always returns
    an empty transcript, never raises, matching how a real degraded
    backend would fail closed rather than crash the pipeline."""

    model_version = "dead-stt-simulated-outage"

    def transcribe(self, pcm: bytes):
        from stt import Transcript

        return Transcript(text="", model_version=self.model_version)


outage_udk = enroll("U_outage", "flamingo skates across the parking lot").udk
outage_journey, _ = api._store.create_or_get("J_outage_test", "U_outage", DeadSTT(), outage_udk)
outage_journey.vad = SingleFixedSegmentVAD(close_at_bytes=16)
outage_journey.kws = MockKWS()
outage_pcm = b"F" * 16
outage_journey.kws.register(outage_pcm, outage_udk.udk_id, distance=0.2)

api._ingest_frame(outage_journey, 0, outage_pcm, api._audio_store)
events_during_outage = api._ingest_frame(outage_journey, 1, b"\x00" * 4, api._audio_store)

check(
    "STT returned nothing for every segment, yet a detection still fired via KWS",
    len(events_during_outage) == 1 and events_during_outage[0]["udk_id"] == outage_udk.udk_id,
    str(events_during_outage),
)
check("the event's transcript is honestly empty (STT really did fail)", events_during_outage[0]["transcript"] == "")
check(
    "model_version.kws reflects the real KWS backend used, not null",
    events_during_outage[0]["model_version"]["kws"] == "mock-kws-0",
    str(events_during_outage[0]["model_version"]),
)

print("=== vectorized DTW == original cell-by-cell DTW (exact) ===")

import numpy as np

from kws import _dtw_distance


def _dtw_reference(a, b):
    """The original O(n*m) Python loop, kept only as the oracle."""
    n, m = len(a), len(b)
    cost = 1.0 - a @ b.T
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m] / max(n, m)


rng = np.random.default_rng(0)
shapes = [(1, 1), (1, 7), (7, 1), (5, 5), (13, 40), (104, 250), (250, 104)]
diffs = []
for n, m in shapes:
    a = rng.normal(size=(n, 16)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(m, 16)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    diffs.append(abs(_dtw_distance(a, b) - _dtw_reference(a, b)))
check(f"identical on {len(shapes)} shapes incl. 1-frame edges and real sizes (max diff {max(diffs):.1e})", max(diffs) == 0.0)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
