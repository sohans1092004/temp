"""
scream_detector.py checks: the pure-signal-processing acoustic distress
signal, and its corroboration-only wiring into udk_engine.py's decide().

Real calibration (calibrate_scream_detector.py, against RAVDESS real
human fear/angry speech and this project's own real recordings) found
only 57.8% recall / 9.0% FPR standalone -- nowhere near good enough to
be a trigger on its own. The one thing that MUST be true, and is what
this file spends most of its checks proving, is that a high
scream_score can never turn a NO_ACTION into anything else -- it can
only add confidence once STT/KWS already independently produced a
TRIGGER_VERIFY-or-better match, the same corroboration-only role
KWS_CORROBORATION_MAX_DISTANCE already plays for STT+KWS agreement.
"""

from __future__ import annotations

import numpy as np

from scream_detector import SAMPLE_RATE, SCREAM_SCORE_THRESHOLD, ScreamDetector
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


def _tone_pcm(freq: float, duration_s: float, am_freq: float | None = None, amplitude: float = 0.5) -> bytes:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    if am_freq is not None:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t)
        carrier = carrier * envelope
    return (np.clip(carrier * amplitude, -1, 1) * 32767).astype("<i2").tobytes()


print("=== ScreamDetector's DSP features behave as claimed (synthetic, deterministic) ===")

detector = ScreamDetector()

smooth = _tone_pcm(300, 1.0)
rough = _tone_pcm(300, 1.0, am_freq=80.0)  # 80Hz AM -- inside the real scream "roughness" band (30-150Hz)
check(
    "an envelope amplitude-modulated inside the roughness band scores higher than a smooth tone at the same pitch",
    detector.score(rough) > detector.score(smooth),
    f"smooth={detector.score(smooth):.3f} rough={detector.score(rough):.3f}",
)

low_pitch = _tone_pcm(150, 1.0)  # inside normal conversational F0 range
high_pitch = _tone_pcm(500, 1.0)  # well above NORMAL_F0_MAX_HZ (255)
check(
    "a pitch well above normal conversational range scores higher than one inside it",
    detector.score(high_pitch) >= detector.score(low_pitch),
    f"low={detector.score(low_pitch):.3f} high={detector.score(high_pitch):.3f}",
)

check("silence scores exactly 0.0 (no divide-by-zero, no spurious signal)", detector.score(b"\x00\x00" * SAMPLE_RATE) == 0.0)
check("empty PCM scores exactly 0.0 (no crash on a zero-length segment)", detector.score(b"") == 0.0)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== decide() never fires on scream_score alone -- corroboration-only, by design ===")

udks = [UDK("UDK_X", "help me please", "GENERAL")]
engine = UDKEngine(udks)

no_text_match = engine.decide("completely unrelated ordinary sentence", now_s=0.0, scream_score=1.0)
check(
    "a maxed-out scream_score (1.0) with NO text/KWS match at all still produces NO_ACTION",
    no_text_match.decision == "NO_ACTION",
    str(no_text_match),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== decide() DOES use scream_score to corroborate an already-partial match ===")

verify_udk = [UDK("UDK_Y", "he's going to hurt me", "GENERAL", verify_by_default=True)]

engine_verify = UDKEngine(verify_udk)
without_scream = engine_verify.decide("he's going to hurt me", now_s=0.0)
check(
    "a verify-by-default phrase with no corroboration stays at TRIGGER_VERIFY",
    without_scream.decision == "TRIGGER_VERIFY",
    str(without_scream),
)

engine_verify_scream = UDKEngine(verify_udk)
with_scream_below = engine_verify_scream.decide("he's going to hurt me", now_s=0.0, scream_score=SCREAM_SCORE_THRESHOLD - 0.05)
check(
    "a scream_score BELOW threshold does not corroborate -- still TRIGGER_VERIFY",
    with_scream_below.decision == "TRIGGER_VERIFY",
    str(with_scream_below),
)

engine_verify_scream2 = UDKEngine(verify_udk)
with_scream_above = engine_verify_scream2.decide("he's going to hurt me", now_s=0.0, scream_score=SCREAM_SCORE_THRESHOLD + 0.05)
check(
    "a scream_score AT/ABOVE threshold corroborates a verify-by-default match into TRIGGER_ALL",
    with_scream_above.decision == "TRIGGER_ALL" and with_scream_above.reason == "acoustic distress signal corroborated the match",
    str(with_scream_above),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== an ordinary (non-verify-by-default) match strictly between the two thresholds also gets the escalation ===")


class FakeSemantic:
    """Same test double as test_semantic.py -- a controlled, deterministic
    confidence instead of depending on real fuzzy-ratio math landing in
    the exact TRIGGER_VERIFY/TRIGGER_ALL gap this check needs."""

    def __init__(self) -> None:
        self._scores: dict[tuple[str, str], float] = {}

    def register(self, a: str, b: str, score: float) -> None:
        self._scores[(a, b)] = score

    def similarity(self, a: str, b: str) -> float:
        return self._scores.get((a, b), 0.0)


mid_udk = [UDK("UDK_Z", "somebody is following me right now", "GENERAL")]
fake_mid = FakeSemantic()
fake_mid.register("somebody is following me right now", "someone is behind me", 0.70)  # strictly between 0.60 and 0.85

engine_mid = UDKEngine(mid_udk, semantic_matcher=fake_mid)
mid_no_scream = engine_mid.decide("someone is behind me", now_s=0.0)
check(
    "sanity: a controlled 0.70 confidence lands at TRIGGER_VERIFY (between thresholds), not TRIGGER_ALL",
    mid_no_scream.decision == "TRIGGER_VERIFY" and mid_no_scream.confidence == 0.70,
    str(mid_no_scream),
)

engine_mid2 = UDKEngine(mid_udk, semantic_matcher=fake_mid)
mid_with_scream = engine_mid2.decide("someone is behind me", now_s=0.0, scream_score=0.9)
check(
    "the same 0.70-confidence match escalates to TRIGGER_ALL once a strong scream_score corroborates it",
    mid_with_scream.decision == "TRIGGER_ALL" and mid_with_scream.reason == "acoustic distress signal corroborated the match",
    str(mid_with_scream),
)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
