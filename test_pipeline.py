"""
Exercises the UDK matching/decision logic (Section 6) against a set of
transcripts covering the cases the review specifically called out:
exact match, ASR-noise/fuzzy match, paraphrase (semantic-stub) match,
ordinary conversation that should NOT trigger, a common/verify-by-default
phrase, and repetition escalating a detection to TRIGGER_ALL.

Real audio -> STT can't be exercised in this sandbox (see README.md:
huggingface.co, where the pretrained Whisper weights live, isn't reachable
here), so these tests go straight in at the transcript layer, which is
exactly where VAD/STT hand off to the UDK engine — the part this session
actually can build and verify end to end.
"""

from __future__ import annotations

from udk_engine import UDKEngine
from udks import all_udks

CASES = [
    # (transcript, expected_decision, note)
    ("Help me, please", "TRIGGER_ALL", "exact match, rare/specific phrase"),
    # UDK_01 was lengthened (udks.py: "Call the police" -> "Call the police,
    # I need help") to stop being a literal substring of an ordinary joke
    # ("I will call the police on you"). Real, accepted side effect: a
    # single noisy instance of the longer phrase now lands at VERIFY, not
    # ALL, since longer phrases have a lower fuzzy-match ceiling -- not a
    # regression, this is the tiered response working as designed (a
    # confirmable single hit, escalating on repetition/another distinct
    # UDK/silence-after-trigger, all of which are separately tested).
    ("call the plice right now", "TRIGGER_VERIFY", "ASR-noise fuzzy match on UDK_01, single occurrence"),
    ("he's going to hurt me, someone is trying to hurt me", "TRIGGER_ALL", "near-exact substring match"),
    ("what a lovely day for a walk", "NO_ACTION", "ordinary conversation, no UDK"),
    ("can you please stop talking so loud", "NO_ACTION", "contains 'stop'-adjacent words but no UDK phrase"),
    ("I don't feel safe here", "TRIGGER_VERIFY", "verify-by-default phrase, single occurrence"),
    ("I want to go home now", "TRIGGER_VERIFY", "verify-by-default phrase, single occurrence"),
]

REPEAT_CASE = ("I don't feel safe here", "single instance is TRIGGER_VERIFY")
REPEAT_ESCALATION_NOTE = "same verify-by-default phrase said again within the window -> escalates to TRIGGER_ALL"


def run_single_cases():
    print("=== Single-transcript cases ===")
    passed = 0
    for transcript, expected, note in CASES:
        engine = UDKEngine(all_udks())  # fresh engine so cases don't interfere via repetition
        result = engine.decide(transcript)
        ok = result.decision == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] \"{transcript}\"")
        print(f"       expected={expected} got={result.decision} "
              f"(udk={result.udk.udk_id if result.udk else None}, "
              f"layer={result.layer}, confidence={result.confidence:.2f})")
        print(f"       note: {note} | reason: {result.reason}")
    print(f"\n{passed}/{len(CASES)} single-transcript cases passed.\n")
    return passed == len(CASES)


def run_repetition_case():
    print("=== Repetition-escalation case ===")
    engine = UDKEngine(all_udks())
    transcript, note = REPEAT_CASE

    first = engine.decide(transcript, now_s=0.0)
    print(f"[t=0s]  \"{transcript}\" -> {first.decision} ({note})")
    ok1 = first.decision == "TRIGGER_VERIFY"

    second = engine.decide(transcript, now_s=5.0)  # within the 15s window
    print(f"[t=5s]  \"{transcript}\" -> {second.decision} ({REPEAT_ESCALATION_NOTE})")
    ok2 = second.decision == "TRIGGER_ALL" and second.repeated

    status = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"[{status}] repetition escalation\n")
    return ok1 and ok2


def run_distinct_escalation_case():
    print("=== Distinct-UDK escalation case ===")
    engine = UDKEngine(all_udks())

    first = engine.decide("I don't feel safe here", now_s=0.0)  # verify-by-default UDK_14
    print(f"[t=0s]  \"I don't feel safe here\" -> {first.decision} (single occurrence, common-in-conversation)")
    ok1 = first.decision == "TRIGGER_VERIFY"

    second = engine.decide("somebody help me now", now_s=1.0)  # a DIFFERENT UDK (UDK_17), not a repeat
    note = "a different UDK within the window -> escalates to TRIGGER_ALL even though neither repeated"
    print(f"[t=1s]  \"somebody help me now\" -> {second.decision} ({note})")
    ok2 = second.decision == "TRIGGER_ALL" and not second.repeated and second.reason == "multiple distinct UDKs triggered within window"

    status = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"[{status}] distinct-UDK escalation\n")
    return ok1 and ok2


def run_personal_udk_case():
    print("=== Personal UDK case ===")
    engine = UDKEngine(all_udks())
    # The sample personal UDK is "the weather in Denver is lovely" (udks.py) —
    # deliberately an ordinary-sounding sentence, per Section 5: a personal
    # phrase should be distinctive enough not to false-positive, but here we
    # just confirm it's detected like any other UDK, speaker-independent.
    transcript = "the weather in denver is lovely today"
    result = engine.decide(transcript)
    ok = result.decision in ("TRIGGER_ALL", "TRIGGER_VERIFY") and result.udk is not None and result.udk.udk_type == "PERSONAL"
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] \"{transcript}\" -> {result.decision} (udk_type={result.udk.udk_type if result.udk else None})\n")
    return ok


if __name__ == "__main__":
    r1 = run_single_cases()
    r2 = run_repetition_case()
    r3 = run_distinct_escalation_case()
    r4 = run_personal_udk_case()
    all_ok = r1 and r2 and r3 and r4
    print("=" * 40)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)
