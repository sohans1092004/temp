"""
separation.py checks: the speech-separation NO_ACTION fallback wired
into api.py/evaluate_pipeline_corpus.py. Fast and portable via
MockSTT + a fake separator -- the real SepformerSeparator was verified
manually this session against 10 real overlapping-speech failures
reproduced from evaluate_pipeline_corpus.py (3/10 recovered, 1 new false
positive out of 20 single-speaker negatives; see separation.py's module
docstring), so it isn't part of the routine automated suite, same call
kws.py/semantic.py's real backends made.
"""

from __future__ import annotations

from separation import retry_with_separation
from stt import MockSTT
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


class FakeSeparator:
    """Test double: returns a fixed list of already-known pcm streams
    instead of running the real SepFormer model."""

    def __init__(self, streams: list[bytes]):
        self._streams = streams

    def separate(self, pcm: bytes) -> list[bytes]:
        return self._streams


class FakeSemantic:
    """Same test double as test_semantic.py -- controllable confidence,
    used here to reproduce the cross-stream distinct-UDK bug without
    depending on real rapidfuzz scores landing in a specific range."""

    def __init__(self):
        self._scores: dict[tuple[str, str], float] = {}

    def register(self, a: str, b: str, score: float) -> None:
        self._scores[(a, b)] = score

    def similarity(self, a: str, b: str) -> float:
        return self._scores.get((a, b), 0.0)


udks = [UDK("UDK_X", "help me please", "GENERAL"), UDK("UDK_Y", "call the police right now", "GENERAL")]

print("=== retry_with_separation recovers a match one of the streams contains ===")

stt = MockSTT()
stream_a, stream_b = b"stream-a", b"stream-b"
stt.register(stream_a, "what time is the meeting")  # unrelated speaker, no UDK
stt.register(stream_b, "help me please")  # the UDK, isolated by separation
engine = UDKEngine(udks)
recovered = retry_with_separation(b"mixed", stt, engine, FakeSeparator([stream_a, stream_b]), now_s=0.0)
check(
    "a stream containing the UDK is found and returned",
    recovered is not None and recovered.udk is not None and recovered.udk.udk_id == "UDK_X",
    str(recovered),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== retry_with_separation returns None when no stream recovers anything ===")

stt2 = MockSTT()
stt2.register(stream_a, "what time is the meeting")
stt2.register(stream_b, "I love this restaurant")
engine2 = UDKEngine(udks)
none_result = retry_with_separation(b"mixed", stt2, engine2, FakeSeparator([stream_a, stream_b]), now_s=0.0)
check("no stream matches -> caller keeps its own NO_ACTION", none_result is None, str(none_result))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== retry_with_separation prefers TRIGGER_ALL over TRIGGER_VERIFY across streams ===")

stt3 = MockSTT()
verify_stream, all_stream = b"verify-stream", b"all-stream"
stt3.register(verify_stream, "call the plice right now")  # ASR-noise fuzzy match, single occurrence -> VERIFY
stt3.register(all_stream, "help me please")  # exact match -> ALL
engine3 = UDKEngine(udks)
best = retry_with_separation(b"mixed", stt3, engine3, FakeSeparator([verify_stream, all_stream]), now_s=0.0)
check("TRIGGER_ALL wins over TRIGGER_VERIFY even when found second", best is not None and best.decision == "TRIGGER_ALL", str(best))

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== a NO_ACTION-only retry never pollutes the engine's repetition window ===")

engine4 = UDKEngine(udks)
stt4 = MockSTT()
stt4.register(stream_a, "what time is the meeting")
stt4.register(stream_b, "I love this restaurant")
retry_with_separation(b"mixed", stt4, engine4, FakeSeparator([stream_a, stream_b]), now_s=0.0)
# A genuine first occurrence right after should still be a single TRIGGER_VERIFY-class
# result (verify-by-default UDKs aside), not already counted as a repeat.
first_real = engine4.decide("help me please", now_s=1.0)
check(
    "engine state untouched by the failed retry (this is the first real occurrence, not treated as a repeat)",
    not first_real.repeated,
    str(first_real),
)

print("=== two separated streams independently matching DIFFERENT UDKs never spuriously escalate (real bug found and fixed) ===")

fake_semantic = FakeSemantic()
fake_semantic.register("help me please", "somebody needs assistance", 0.65)  # VERIFY-range alone
fake_semantic.register("call the police right now", "get the authorities involved", 0.70)  # VERIFY-range alone
stream_x, stream_y = b"stream-x", b"stream-y"
stt5 = MockSTT()
stt5.register(stream_x, "somebody needs assistance")
stt5.register(stream_y, "get the authorities involved")
engine5 = UDKEngine(udks, semantic_matcher=fake_semantic)
result = retry_with_separation(b"mixed", stt5, engine5, FakeSeparator([stream_x, stream_y]), now_s=0.0)
check(
    "two simultaneous stream guesses on different UDKs don't force TRIGGER_ALL via spurious distinct-UDK escalation",
    result is not None and result.decision == "TRIGGER_VERIFY" and not getattr(result, "repeated", False),
    str(result),
)
check(
    "the returned decision's own confidence is one real candidate's score, not inflated by escalation",
    result is not None and result.confidence in (0.65, 0.70),
    str(result),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== WeSepTargetSeparator: only a self-verified extraction reaches the retry ===")

from wesep_extraction import WeSepTargetSeparator


class FakeExtractor:
    def extract(self, mixture_pcm: bytes, enrollment_pcm: bytes) -> bytes:
        return b"extracted:" + mixture_pcm


class FakeVerifier:
    def __init__(self, score: float):
        self._score = score

    def score(self, enrollment_pcm: bytes, pcm: bytes) -> float:
        return self._score


accept = WeSepTargetSeparator(b"enroll", extractor=FakeExtractor(), verifier=FakeVerifier(0.60))
check("extraction scoring above threshold is returned as the single stream", accept.separate(b"mix") == [b"extracted:mix"])
reject = WeSepTargetSeparator(b"enroll", extractor=FakeExtractor(), verifier=FakeVerifier(0.30))
check("extraction scoring below threshold is discarded (user absent / failed extraction)", reject.separate(b"mix") == [])
stt6 = MockSTT()
stt6.register(b"extracted:mix", "help me please")
check(
    "accepted extraction flows through retry_with_separation to a real detection",
    (r := retry_with_separation(b"mix", stt6, UDKEngine(udks), accept, now_s=0.0)) is not None and r.udk.udk_id == "UDK_X",
)
check("rejected extraction leaves the caller's NO_ACTION alone",
      retry_with_separation(b"mix", stt6, UDKEngine(udks), reject, now_s=0.0) is None)

print("=== recheck_verify_with_separation: a weak wrong-UDK VERIFY yields to the user's exact UDK, counted once ===")

from separation import recheck_verify_with_separation

sem7 = FakeSemantic()
sem7.register("call the police right now", "can you help me with this", 0.70)  # other person's sentence, weak semantic VERIFY
engine7 = UDKEngine(udks, semantic_matcher=sem7)
state7 = engine7.snapshot()
primary7 = engine7.decide("can you help me with this", now_s=0.0)
stt7 = MockSTT()
stt7.register(b"extracted:mix", "help me please")  # the user's own extracted voice, exact UDK_X
fixed = recheck_verify_with_separation(b"mix", primary7, state7, stt7, engine7, accept, now_s=0.0)
check("setup: primary is a semantic VERIFY on UDK_Y", primary7.decision == "TRIGGER_VERIFY" and primary7.udk.udk_id == "UDK_Y")
check("replaced by the exact UDK_X from the extracted voice", fixed.udk.udk_id == "UDK_X" and fixed.layer == "exact", str(fixed))
check("the replaced guess is NOT counted: engine saw one UDK for this segment, no distinct-UDK escalation",
      len(engine7._recent_distinct) == 1, str(engine7._recent_distinct))

engine8 = UDKEngine(udks, semantic_matcher=sem7)
state8 = engine8.snapshot()
primary8 = engine8.decide("can you help me with this", now_s=0.0)
stt8 = MockSTT()
stt8.register(b"extracted:mix", "call the police right now")  # extraction agrees with primary -> nothing to change
check("same UDK from extraction -> original decision kept",
      recheck_verify_with_separation(b"mix", primary8, state8, stt8, engine8, accept, now_s=0.0) is primary8)
stt9 = MockSTT()
stt9.register(b"extracted:mix", "somebody needs assistance")  # extraction gives nothing exact -> never vetoes
check("non-exact extraction -> original decision kept (never a veto)",
      recheck_verify_with_separation(b"mix", primary8, state8, stt9, engine8, accept, now_s=0.0) is primary8)

print("=== separated_extra_event: WeSep still runs when a sentence already fired, never double-counts ===")

from separation import separated_extra_event

udks10 = udks + [UDK("UDK_Z", "get away from me", "GENERAL")]
engine10 = UDKEngine(udks10)
# The real case ("You do me!" in the pocket clip): the whole segment matched
# nothing, one sentence of it fired on its own. Built directly.
primary10 = engine10.decide("what was going anywhere", now_s=0.0)
extras10 = [engine10.decide("help me please", now_s=0.0)]
check("setup: whole segment NO_ACTION, a sentence fired UDK_X",
      primary10.decision == "NO_ACTION" and extras10[0].udk.udk_id == "UDK_X")
stt10 = MockSTT()
stt10.register(b"extracted:mix", "get away from me")  # user's own extracted voice: a DIFFERENT, exact UDK
added = separated_extra_event(b"mix", primary10, extras10, stt10, engine10, accept, now_s=0.0)
check("different exact UDK from the user's voice is added", added is not None and added.udk.udk_id == "UDK_Z", str(added))

engine11 = UDKEngine(udks10)
primary11 = engine11.decide("what was going anywhere", now_s=0.0)
extras11 = [engine11.decide("help me please", now_s=0.0)]
stt11 = MockSTT()
stt11.register(b"extracted:mix", "help me please")  # same UDK the mixture already fired
check("same UDK already fired in this segment -> not added (no double count)",
      separated_extra_event(b"mix", primary11, extras11, stt11, engine11, accept, now_s=0.0) is None)
check("rejected extraction (user absent) -> nothing added",
      separated_extra_event(b"mix", primary11, extras11, stt10, UDKEngine(udks10), reject, now_s=0.0) is None)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
