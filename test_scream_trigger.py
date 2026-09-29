"""Scream-alone trigger (scream_trigger.py + api.py wiring), with a fake BEATs detector
so no model loads. python test_scream_trigger.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import api
import scream_trigger as ST
from db import JourneyDB
from event_delivery import EventDeliveryQueue
from storage import AudioStore
from stt import MockSTT
from udks import SAMPLE_PERSONAL_UDK
from vad import SpeechSegment

_tmp = Path(tempfile.mkdtemp(prefix="udk_scream_test_"))
api._audio_store = AudioStore(_tmp / "audio")
api._event_queue = EventDeliveryQueue(_tmp / "events")
api._db = JourneyDB(_tmp / "journeys.sqlite3")
api._store = api.JourneyStore(db=api._db)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    passed, failed = passed + cond, failed + (not cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {'' if cond else detail}")


class FakeBEATs:
    """label_probs: 'Screaming' (index 34) = the PCM's first sample value / 1000, so a test
    controls the score per window by the bytes it passes in."""
    model_version = "fake-beats"

    def label_probs(self, pcm):
        p = np.zeros(527, np.float32)
        p[34] = np.frombuffer(pcm[:2], dtype="<i2")[0] / 1000 if len(pcm) >= 2 else 0
        return p

    def score(self, pcm):
        return 0.0


def pcm_with(levels):
    """One value per 0.5 s hop -> each 2 s window starts with that hop's value."""
    return b"".join(np.full(ST.HOP, int(l * 1000), dtype="<i2").tobytes() for l in levels)


det = FakeBEATs()
# 3 s of audio = 3 windows (starts 0, 0.5, 1.0 s)
check("fires on 2 consecutive windows >= 0.10", ST.check(det, pcm_with([0.2, 0.2, 0.2, 0, 0, 0]))[0])
check("does NOT fire on a single window", not ST.check(det, pcm_with([0.2, 0.0, 0.2, 0, 0, 0]))[0])
check("does NOT fire below threshold", not ST.check(det, pcm_with([0.05] * 6))[0])
fired, peak, cls = ST.check(det, pcm_with([0.3, 0.4, 0, 0, 0, 0]))
check("reports peak score and class", fired and abs(peak - 0.4) < 1e-6 and cls == "Screaming", (fired, peak, cls))


class FakeWailOnly(FakeBEATs):  # the class that caused 79 of 182 real YouTube false alarms
    def label_probs(self, pcm):
        p = np.zeros(527, np.float32)
        p[431] = 0.9  # "Wail, moan"
        return p


check("a strong Wail/moan-only sound does NOT fire (dropped class)", not ST.check(FakeWailOnly(), pcm_with([0.5] * 6))[0])
check("live rule = Screaming, Crying, Whimper, Children shouting",
      set(ST.CLASSES.values()) == {"Screaming", "Crying, sobbing", "Whimper", "Children shouting"}, str(ST.CLASSES))

# ---- live loop: a scream with NO words -> TRIGGER_VERIFY; flag off -> nothing
seg_pcm = pcm_with([0.5, 0.5, 0.5, 0, 0, 0])


class OneSegmentVAD:
    def segment_speech(self, pcm):
        return [SpeechSegment(start_ms=0, end_ms=3000, pcm=pcm[: len(seg_pcm)])] if len(pcm) >= len(seg_pcm) else []

    def total_ms(self, pcm):
        return len(pcm) // 32


def run(jid, trigger_on, words=""):
    api._scream_detector, api._scream_trigger_enabled = det, trigger_on
    api._stt_backend = MockSTT()
    api._stt_backend.register(seg_pcm, words)
    api._parakeet_stt = None
    j, _ = api._store.create_or_get(jid, "U1", api._stt_backend, SAMPLE_PERSONAL_UDK)
    j.vad = OneSegmentVAD()
    return api._ingest_frame(j, 0, seg_pcm + b"\x00\x00" * 100, api._audio_store)


ev = run("J_scream_on", True)
check("scream + no words, trigger ON -> one TRIGGER_VERIFY (acoustic)",
      len(ev) == 1 and ev[0]["decision"] == "TRIGGER_VERIFY" and ev[0]["udk_id"] is None, str(ev))
check("flag OFF -> the same scream raises nothing", run("J_scream_off", False) == [])
ev = run("J_scream_udk", True, words="get away from me")
check("a UDK alert on the segment takes precedence (no extra acoustic event)",
      len(ev) == 1 and ev[0]["udk_id"] == "UDK_03", str(ev))

api._scream_trigger_enabled, api._scream_detector = False, None
print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))
if failed:
    raise SystemExit(1)
