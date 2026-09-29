"""dual_asr_guard.py + udk_engine.py's dual_asr_disagreement wiring
checks. Fast and portable via a fake double for the engine-level wiring
(same FakeSeparator/FakeSemantic pattern as test_separation.py/
test_semantic.py) -- the real Wav2Vec2DisagreementGuard backend was
verified manually this session against 3 real false-UDK-trigger cases
found this session (see dual_asr_guard.py's module docstring for the
transcripts/scores), same call this project's other real backends made
(kws.py's Wav2Vec2DTWSpotter, stt.py's FasterWhisperSTT)."""

from __future__ import annotations

from rapidfuzz import fuzz

from dual_asr_guard import AGREEMENT_THRESHOLD
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


print("=== AGREEMENT_THRESHOLD against this session's 3 real validated false-trigger cases ===")

# (primary whisper transcript, wav2vec2 transcript) -- real values from
# dual_asr_guard.py's module docstring, all 3 real cases that produced a
# false UDK trigger this session.
real_cases = [
    ("i think it's time for us to go back home.", "", "clip_04 ch_a"),
    ("we should order more category of me.", "we shon't don't da won't coa me", "clip_08 mono"),
    ("we should all be more careful of me.", "westo do daoaa", "clip_08 ch_b"),
]
for primary, secondary, label in real_cases:
    agreement = fuzz.token_set_ratio(primary, secondary) / 100.0
    check(
        f"{label}: real disagreement ({agreement:.3f}) is below AGREEMENT_THRESHOLD ({AGREEMENT_THRESHOLD})",
        agreement < AGREEMENT_THRESHOLD,
        f"agreement={agreement:.3f}",
    )

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== wired into UDKEngine.decide(): dual_asr_disagreement vetoes a text-only match ===")

udks = [UDK("UDK_X", "let go of me", "GENERAL")]

engine = UDKEngine(udks)
agrees_result = engine.decide("let go of me", now_s=0.0, dual_asr_disagreement=False)
check(
    "an exact match with dual_asr_disagreement=False fires normally",
    agrees_result.decision != "NO_ACTION",
    str(agrees_result),
)

engine2 = UDKEngine(udks)
disagrees_result = engine2.decide("let go of me", now_s=0.0, dual_asr_disagreement=True)
check(
    "the SAME exact match with dual_asr_disagreement=True is vetoed to NO_ACTION",
    disagrees_result.decision == "NO_ACTION" and "dual-ASR disagreement" in disagrees_result.reason,
    str(disagrees_result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== KWS independence: dual_asr_disagreement must not veto a KWS-corroborated match ===")

from kws import KWSMatch

engine3 = UDKEngine(udks)
kws_match = KWSMatch(phrase_id="UDK_X", distance=0.05)
corroborated_result = engine3.decide("let go of me", now_s=0.0, kws_match=kws_match, dual_asr_disagreement=True)
check(
    "STT+KWS agreement (corroborated) is not vetoed even with dual_asr_disagreement=True",
    corroborated_result.decision != "NO_ACTION",
    str(corroborated_result),
)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
