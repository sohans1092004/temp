"""
semantic.py checks: the optional real semantic-matching backend wired
into udk_engine.py's layer 3. Fast and portable via a fake backend --
the real SentenceTransformerSemanticMatcher was verified manually this
session against 5 real phrase pairs, compared against two other models
before picking paraphrase-multilingual-MiniLM-L12-v2 (see README.md for
the numbers); it needs sentence-transformers + torch + a model download,
so it isn't part of the routine automated suite, same call kws.py's
Wav2Vec2DTWSpotter and stt.py's FasterWhisperSTT verification made.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from udk_engine import UDKEngine, match_transcript
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


class FakeSemantic:
    """Test double: returns a configured score for a given (a, b) pair
    instead of running a real embedding model."""

    def __init__(self, fuzzy_threshold: float | None = None):
        self._scores: dict[tuple[str, str], float] = {}
        if fuzzy_threshold is not None:
            self.fuzzy_threshold = fuzzy_threshold

    def register(self, a: str, b: str, score: float) -> None:
        self._scores[(a, b)] = score

    def similarity(self, a: str, b: str) -> float:
        return self._scores.get((a, b), 0.0)


udks = [UDK("UDK_X", "he's going to hurt me", "GENERAL")]

print("=== match_transcript falls back to the rapidfuzz stub with no semantic_matcher ===")

no_matcher_result = match_transcript("someone is threatening me", udks)
check(
    "no semantic_matcher given -> match_transcript runs fine without one (whichever layer wins)",
    no_matcher_result.layer in ("exact", "fuzzy", "semantic", "none"),
    str(no_matcher_result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== match_transcript uses a real semantic_matcher when given ===")

fake = FakeSemantic()
fake.register("hes going to hurt me", "someone is threatening me", 0.75)
with_matcher_result = match_transcript("someone is threatening me", udks, semantic_matcher=fake)
check(
    "semantic_matcher's score is used directly, undiscounted",
    with_matcher_result.udk is not None and with_matcher_result.confidence == 0.75 and with_matcher_result.layer == "semantic",
    str(with_matcher_result),
)

fake_below_threshold = FakeSemantic()
fake_below_threshold.register("hes going to hurt me", "completely unrelated text", 0.3)
below_threshold_result = match_transcript("completely unrelated text", udks, semantic_matcher=fake_below_threshold)
check("a semantic score below 0.55 doesn't produce a match", below_threshold_result.udk is None, str(below_threshold_result))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== fuzzy layer's raw gate is per-matcher (Telugu/Kannada/Tamil FPR fix) ===")

strict_udks = [UDK("UDK_Y", "call the police right now", "GENERAL")]

default_gate_result = match_transcript("please call someone later", strict_udks)
check(
    "raw fuzzy ratio below the default 0.55 gate -> no match",
    default_gate_result.udk is None,
    str(default_gate_result),
)

fake_strict = FakeSemantic(fuzzy_threshold=0.90)
lenient_would_match = fuzz.partial_ratio("call the police right now", "call the fire brigade right now") / 100.0
strict_gate_result = match_transcript("call the fire brigade right now", strict_udks, semantic_matcher=fake_strict)
check(
    "a matcher declaring a stricter fuzzy_threshold (0.90) rejects a ratio that would have cleared the default 0.55",
    lenient_would_match >= 0.55 and strict_gate_result.udk is None,
    f"raw_ratio={lenient_would_match:.2f} result={strict_gate_result}",
)

fake_no_fuzzy_threshold = FakeSemantic()
backward_compat_result = match_transcript("please call someone later", strict_udks, semantic_matcher=fake_no_fuzzy_threshold)
check(
    "a matcher with no fuzzy_threshold attribute falls back to the 0.55 default (backward compat)",
    backward_compat_result.udk is None,
    str(backward_compat_result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== decide_all: every UDK said in one segment is reported, not just the best one ===")

from udk_engine import UDKEngine

two_udks = [UDK("UDK_A", "let go of me", "GENERAL"), UDK("UDK_B", "please don't hurt me", "GENERAL")]
multi = UDKEngine(two_udks).decide_all("Let go of me! Just calm down! Please don't hurt me!", now_s=0.0)
check("two different UDKs in separate sentences -> two events", sorted(e.udk.udk_id for e in multi) == ["UDK_A", "UDK_B"], str(multi))
check("first event is the whole-segment decision (same as decide())",
      multi[0].udk.udk_id == UDKEngine(two_udks).decide("Let go of me! Just calm down! Please don't hurt me!", now_s=0.0).udk.udk_id)
single = UDKEngine(two_udks).decide_all("Let go of me!", now_s=0.0)
check("one sentence -> exactly one event", len(single) == 1, str(single))
quiet = UDKEngine(two_udks).decide_all("What a lovely day. Shall we walk?", now_s=0.0)
check("ordinary sentences -> no extra events", len(quiet) == 1 and quiet[0].decision == "NO_ACTION", str(quiet))

short_udks = [UDK("UDK_A", "let go of me", "GENERAL"), UDK("UDK_B", "please don't hurt me", "GENERAL"),
              UDK("UDK_C", "i'm not safe", "GENERAL")]
# Whole segment matches UDK_A, so UDK_B can only come from the short sentence on its own.
shout = UDKEngine(short_udks).decide_all("Let go of me right now. Please don't!", now_s=0.0)
check("short sentence made of a UDK's whole words ('Please don't!') still counts on its own",
      shout[0].udk.udk_id == "UDK_A" and any(e.udk and e.udk.udk_id == "UDK_B" for e in shout[1:]), str(shout))
no_frag = UDKEngine(short_udks).decide_all("We talked about the plan for hours. No.", now_s=0.0)
check("short sentence matching only INSIDE a word ('No.' in 'not') is ignored",
      all(e.udk is None or e.udk.udk_id != "UDK_C" for e in no_frag[1:]), str(no_frag))
go_to = UDKEngine(short_udks).decide_all("We walked to the station together. Go to.", now_s=0.0)
check("short sentence whose words aren't a run in any UDK ('Go to.') is ignored", len(go_to) == 1, str(go_to))
long_ok = UDKEngine(short_udks).decide_all("Calm down now. Let go of me.", now_s=0.0)
check("sentences of 3+ words are matched as before", any(e.udk and e.udk.udk_id == "UDK_A" for e in long_ok), str(long_ok))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== a short fragment ambiguous across multiple UDKs is NOT full confidence (real bug found and fixed) ===")

ambiguous_udks = [
    UDK("UDK_A", "please call the police for help", "GENERAL"),
    UDK("UDK_B", "someone please help me now", "GENERAL"),
]

ambiguous_result = match_transcript("help", ambiguous_udks)
check(
    "a fragment that's a substring of TWO different UDK phrases is capped below full confidence, not an unconfirmed 1.0",
    ambiguous_result.udk is not None and ambiguous_result.layer == "exact" and ambiguous_result.confidence < 1.0,
    str(ambiguous_result),
)
for filler in ("please", "Please, please.", "Right?", "I'm", "now"):
    r = match_transcript(filler, ambiguous_udks)
    check(f"a transcript of only non-distinctive words never matches: {filler!r}", r.udk is None, str(r))

unique_result = match_transcript("police", ambiguous_udks)
check(
    "a fragment that's a substring of only ONE UDK phrase still keeps full confidence (real, unambiguous evidence)",
    unique_result.udk is not None and unique_result.udk.udk_id == "UDK_A" and unique_result.confidence == 1.0,
    str(unique_result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== UDKEngine wires semantic_matcher through decide() ===")

engine = UDKEngine(udks, semantic_matcher=fake)
decision = engine.decide("someone is threatening me", now_s=0.0)
check(
    "UDKEngine.decide() uses the injected semantic_matcher's score (0.75 -> TRIGGER_VERIFY range)",
    decision.udk is not None and decision.udk.udk_id == "UDK_X" and decision.decision == "TRIGGER_VERIFY",
    str(decision),
)

engine_no_matcher = UDKEngine(udks)
decision_no_matcher = engine_no_matcher.decide("someone is threatening me", now_s=0.0)
check("UDKEngine without a semantic_matcher still runs (falls back to the stub)", decision_no_matcher.decision in ("TRIGGER_VERIFY", "TRIGGER_ALL", "NO_ACTION"))

print("=== fuzzy layer needs the UDK's own words in a long sentence (real YouTube false alarms) ===")

from udks import GENERAL_UDKS  # noqa: E402

no_semantic = FakeSemantic()  # scores 0: isolates the fuzzy layer
for text in ("But I'm not saying this only because of the MES statement, I'm saying this because",  # was UDK_02
             "Among the important things that come out of what he says, I want to go to you Richie"):  # was UDK_18
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"long sentence that only starts like a UDK is not a fuzzy match: {text[:40]!r}", r.layer != "fuzzy", str(r))
for text, udk_id in (("please someone help me he's not letting me leave please help me", "UDK_08"),  # word order differs
                     ("call the plice i need help", "UDK_01")):  # ASR noise
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"real danger still matches {udk_id}: {text[:40]!r}", r.udk is not None and r.udk.udk_id == udk_id and r.confidence >= 0.6, str(r))

print("=== fragment rule: a short transcript needs one of the UDK's anchor words ===")

import udk_engine  # noqa: E402

for text in ("Help!", "Police!", "Get away!", "Let go!", "Stay back!", "I'm scared", "Don't hurt me", "Help me",
             "Someone's following me", "call the plice"):
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"short real cry still matches: {text!r}", r.udk is not None and r.confidence >= 0.6, str(r))
# known limit: "Come on now." still matches UDK_18 ("come" is one letter from the anchor "home")
blocked = ("exact", "fuzzy") if udk_engine.FRAGMENT_FUZZY else ("exact",)
for text in ("I don't", "I want", "I don't know.", "No, I don't know.", "Now, I want to", "You want to be here.",
             "He wants to clean.", "F need.", "No need of"):
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"anchorless fragment isn't a {'/'.join(blocked)} match: {text!r}", r.layer not in blocked, str(r))
for text in ("From my father.", "Being honest.", "For real."):  # prepositions/aux are not anchors
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"function-word fragment isn't an exact/fuzzy match: {text!r}", r.layer not in blocked, str(r))
missing = [u.udk_id for u in GENERAL_UDKS
           if not [w for w in udk_engine._normalize(u.phrase).split() if w not in udk_engine.FUNCTION_WORDS]]
check("every standard UDK keeps at least one anchor word", not missing, str(missing))
for text in ("Please don't!", "No, don't!"):
    r = match_transcript(text, GENERAL_UDKS, semantic_matcher=no_semantic)
    check(f"'don't' as a command still matches: {text!r}", r.udk is not None, str(r))
r = match_transcript("You do me!", GENERAL_UDKS, semantic_matcher=no_semantic)  # garbled "Let go of me", prepped pocket clip
check(f"'You do me!' (no anchor) matches only with UDK_FRAGMENT_FUZZY=0 (now {int(udk_engine.FRAGMENT_FUZZY)})",
      (r.udk is not None) != udk_engine.FRAGMENT_FUZZY, str(r))

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
