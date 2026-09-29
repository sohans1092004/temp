"""
api.py's live path, brought in line with the evaluated best configuration
(2026-09-24), with every STT stubbed by MockSTT so no model loads:

  - _either_stt_decisions (UDK_ENABLE_PARAKEET=1) keeps the stronger of the
    Whisper and Parakeet decision lists, and leaves the engine as if ONLY the
    kept list had been decided (two transcripts of one segment must never
    count as a repetition escalation);
  - the live loop still runs the second STT when the primary heard NOTHING
    (it used to skip such segments before any second opinion could run);
  - the second STT is only used while the journey is on the English backend;
  - per-sentence matching (decide_all): two UDKs in one segment -> two events.

python test_parakeet_either.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import api
from db import JourneyDB
from event_delivery import EventDeliveryQueue
from stt import MockSTT
from storage import AudioStore
from udk_engine import UDKEngine
from udks import GENERAL_UDKS, SAMPLE_PERSONAL_UDK
from vad import SpeechSegment

_tmp_root = Path(tempfile.mkdtemp(prefix="udk_parakeet_test_"))
api._audio_store = AudioStore(_tmp_root / "audio")
api._event_queue = EventDeliveryQueue(_tmp_root / "events")
api._db = JourneyDB(_tmp_root / "journeys.sqlite3")
api._store = api.JourneyStore(db=api._db)

passed = failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    passed, failed = passed + condition, failed + (not condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {'' if condition else detail}")


print("=== _either_stt_decisions ===")
PHRASE = "get away from me"
now = time.monotonic()
either = api._either_stt_decisions

engine = UDKEngine(GENERAL_UDKS)
pre = engine.snapshot()
primary = engine.decide_all("", now_s=now)  # Whisper heard nothing
kept = either(engine, primary, pre, PHRASE, now, None, None)
check("primary NO_ACTION + second hears the UDK -> the second's alert is kept",
      kept[0].decision != "NO_ACTION" and kept[0].udk is not None and kept[0].udk.udk_id == "UDK_03", str(kept))
reference = UDKEngine(GENERAL_UDKS)
reference.decide_all(PHRASE, now_s=now)
check("engine state == as if ONLY the kept decisions were made", engine.snapshot() == reference.snapshot())

engine = UDKEngine(GENERAL_UDKS)
pre = engine.snapshot()
primary = engine.decide_all(PHRASE, now_s=now)  # Whisper heard it
after_primary = engine.snapshot()
kept = either(engine, primary, pre, "we are out of milk again", now, None, None)
check("primary alert + second hears nothing useful -> primary kept", kept is primary)
check("engine state restored to the primary's (the second guess left no trace)", engine.snapshot() == after_primary)

engine = UDKEngine(GENERAL_UDKS)
pre = engine.snapshot()
primary = engine.decide_all(PHRASE, now_s=now)
either(engine, primary, pre, PHRASE, now, None, None)  # both hear it
reference = UDKEngine(GENERAL_UDKS)
reference.decide_all(PHRASE, now_s=now)
check("both paths hearing the same phrase counts ONCE, not as a repeat", engine.snapshot() == reference.snapshot())

engine = UDKEngine(GENERAL_UDKS)
primary = engine.decide_all("", now_s=now)
check("empty second transcript -> primary returned untouched",
      either(engine, primary, engine.snapshot(), "", now, None, None) is primary)

engine = UDKEngine(GENERAL_UDKS)
pre = engine.snapshot()
primary = engine.decide_all("nobody is going anywhere", now_s=now)
kept = either(engine, primary, pre, "Let go of me! Please don't hurt me!", now, None, None)
check("the second transcript's per-sentence extras come along with it",
      {d.udk.udk_id for d in kept if d.udk} >= {"UDK_10", "UDK_13"}, str([(d.decision, d.udk and d.udk.udk_id) for d in kept]))


print("\n=== live loop (_ingest_frame) ===")


class FixedBlockVAD:
    def __init__(self, block: int):
        self.block = block

    def segment_speech(self, pcm):
        return [SpeechSegment(start_ms=i * self.block, end_ms=(i + 1) * self.block, pcm=pcm[i * self.block:(i + 1) * self.block])
                for i in range(len(pcm) // self.block)]

    def total_ms(self, pcm):
        return (len(pcm) // self.block) * self.block


BLOCK = 8
block_b = b"B" * BLOCK


def run_journey(journey_id: str, second: MockSTT | None, stt=None) -> list[dict]:
    api._parakeet_stt = second
    journey, _ = api._store.create_or_get(journey_id, "U1", stt or api._stt_backend, SAMPLE_PERSONAL_UDK)
    journey.vad = FixedBlockVAD(BLOCK)
    events = []
    for seq, block in enumerate((b"A" * BLOCK, block_b, b"C" * BLOCK)):
        events += api._ingest_frame(journey, seq, block, api._audio_store)
    return events


api._stt_backend = MockSTT()  # primary (Whisper stand-in) hears nothing in any block
parakeet = MockSTT()
parakeet.register(block_b, PHRASE)

events = run_journey("J_whisper_deaf_no_parakeet", None)
check("Whisper-only: a segment Whisper heard nothing in raises no alert", events == [], str(events))

events = run_journey("J_whisper_deaf_parakeet_hears", parakeet)
check("Whisper + Parakeet: the same segment now alerts (Parakeet heard the UDK)",
      len(events) == 1 and events[0]["udk_id"] == "UDK_03", str(events))

other_language_stt = MockSTT()  # e.g. an Indic backend: not the English _stt_backend
events = run_journey("J_non_english", parakeet, stt=other_language_stt)
check("second STT is NOT used when the journey isn't on the English backend", events == [], str(events))

print("\n=== per-sentence matching in the live loop (decide_all) ===")
api._stt_backend = MockSTT()
api._stt_backend.register(block_b, "Let go of me! Please don't hurt me!")
events = run_journey("J_two_udks_one_segment", None)
check("one segment with two different UDK sentences -> an event for EACH",
      sorted(e["udk_id"] for e in events) == ["UDK_10", "UDK_13"], str([e["udk_id"] for e in events]))
check("their event_ids are distinct (no dedup collision)", len({e["event_id"] for e in events}) == len(events))

api._parakeet_stt = None
print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))
if failed:
    raise SystemExit(1)
