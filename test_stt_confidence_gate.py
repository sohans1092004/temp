"""stt_confidence_gate.py + udk_engine.py's stt_suspect wiring checks.
Ground truth is this session's real measured values (see
stt_confidence_gate.py's module docstring), not synthetic numbers."""

from __future__ import annotations

from stt import Transcript
from stt_confidence_gate import is_suspect
from udk_engine import UDKEngine
from udks import UDK

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


print("=== is_suspect() against this session's real measured cases ===")

check(
    "clip_08 separated-channel fabrication (temperature=1.0) is flagged",
    is_suspect(Transcript(text="x", model_version="t", avg_logprob=-1.227, temperature=1.0)),
)
check(
    "clip_16 repeated-sentence hallucination (temperature=0.2) is flagged",
    is_suspect(Transcript(text="x", model_version="t", avg_logprob=-0.968, temperature=0.2)),
)
check(
    "clip_14 noisy-condition wrong transcript (temperature=1.0) is flagged",
    is_suspect(Transcript(text="x", model_version="t", avg_logprob=-1.250, temperature=1.0)),
)
check(
    "clip_01/09/13 genuine-correct baselines are NOT flagged",
    not any(
        is_suspect(Transcript(text="x", model_version="t", avg_logprob=lp, temperature=0.0))
        for lp in (-0.574, -0.673, -0.472)
    ),
)
check(
    "documented blind spot: clip_04/clip_08 'confidently wrong' cases are NOT flagged",
    not any(
        is_suspect(Transcript(text="x", model_version="t", avg_logprob=lp, temperature=0.0))
        for lp in (-0.733, -0.576, -0.730, -0.920)
    ),
    "if this fails, the module docstring's documented limitation is stale",
)
check(
    "a backend with no confidence metadata (e.g. MockSTT) is never flagged",
    not is_suspect(Transcript(text="x", model_version="mock")),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== wired into UDKEngine.decide(): stt_suspect vetoes a text-only match ===")

udks = [UDK("UDK_X", "let go of me", "GENERAL")]
engine = UDKEngine(udks)

not_suspect_result = engine.decide("let go of me", now_s=0.0, stt_suspect=False)
check(
    "an exact match with stt_suspect=False fires normally",
    not_suspect_result.decision != "NO_ACTION",
    str(not_suspect_result),
)

engine2 = UDKEngine(udks)
suspect_result = engine2.decide("let go of me", now_s=0.0, stt_suspect=True)
check(
    "the SAME exact match with stt_suspect=True is vetoed to NO_ACTION",
    suspect_result.decision == "NO_ACTION" and "STT-confidence guardrail" in suspect_result.reason,
    str(suspect_result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== KWS independence: stt_suspect must not veto a KWS-corroborated or KWS-only match ===")

from kws import KWSMatch

kws_udks = [UDK("UDK_Y", "call the police right now", "GENERAL")]
engine3 = UDKEngine(kws_udks)
kws_match = KWSMatch(phrase_id="UDK_Y", distance=0.05)
corroborated_result = engine3.decide(
    "call the police right now", now_s=0.0, kws_match=kws_match, stt_suspect=True
)
check(
    "STT+KWS agreement (corroborated) is not vetoed even with stt_suspect=True",
    corroborated_result.decision != "NO_ACTION",
    str(corroborated_result),
)

engine4 = UDKEngine(kws_udks)
kws_only_result = engine4.decide("completely unrelated text", now_s=0.0, kws_match=kws_match, stt_suspect=True)
check(
    "a KWS-only match (STT found nothing) is not vetoed by an STT-quality signal",
    kws_only_result.decision != "NO_ACTION" and kws_only_result.layer == "kws",
    str(kws_only_result),
)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
