"""
Ties VAD -> STT (+ optional KWS, Section 4's Approach E) -> UDKEngine
together and prints a decision event for each speech segment, in the
shape of Section 9's event schema (trimmed to what a local console
prototype needs — no journey_id/location/etc. since there's no real
journey session here yet, per Section 14's M1 scope: "Local audio -> VAD
-> STT -> 20 UDKs -> console decision output").
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from kws import KWSBackend
from stt import STTBackend, Transcript
from udk_engine import UDKEngine, DecisionEvent
from vad import VAD, SpeechSegment
from udks import all_udks


def _event_json(event: DecisionEvent, segment: SpeechSegment) -> dict:
    return {
        "decision": event.decision,
        "udk_id": event.udk.udk_id if event.udk else None,
        "udk_type": event.udk.udk_type if event.udk else None,
        "udk_phrase": event.udk.phrase if event.udk else None,
        "confidence": round(event.confidence, 3),
        "match_layer": event.layer,
        "repeated_in_window": event.repeated,
        "reason": event.reason,
        "transcript": event.transcript,
        "segment_start_ms": segment.start_ms,
        "segment_end_ms": segment.end_ms,
    }


def run_pipeline(
    pcm: bytes,
    stt: STTBackend,
    engine: UDKEngine,
    vad: VAD | None = None,
    kws: KWSBackend | None = None,
    scream_detector: object | None = None,  # beats_distress_detector.BEATsDistressDetector -- corroboration-only, see udk_engine.py
) -> list[dict]:
    vad = vad or VAD()
    segments = vad.segment_speech(pcm)
    events = []
    for seg in segments:
        transcript: Transcript = stt.transcribe(seg.pcm)
        kws_match = kws.spot(seg.pcm) if kws is not None else None
        # Deciding even on an empty transcript is what lets KWS fire on
        # its own when STT gets nothing (an outage, Section 3) -- skipping
        # early here the way this used to would silently defeat the
        # entire point of running KWS in parallel.
        if not transcript.text and kws_match is None:
            continue
        scream_score = scream_detector.score(seg.pcm) if scream_detector is not None else None
        decision = engine.decide(transcript.text, now_s=time.monotonic(), kws_match=kws_match, scream_score=scream_score)
        event = _event_json(decision, seg)
        events.append(event)
        if decision.decision != "NO_ACTION":
            print(json.dumps(event, indent=2))
    return events
