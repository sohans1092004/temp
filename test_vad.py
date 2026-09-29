"""VAD.segment_speech checks -- previously untested directly (only
exercised indirectly through fake VAD test doubles elsewhere). Focuses
on the real bug found and fixed against real continuous audio (a
56-second real video clip with no VAD-detectable pause, which stayed one
never-finalizing segment for the clip's entire duration): a continuous
run of speech-classified frames must be capped at max_segment_ms, not
left to grow forever.

Uses a fake underlying webrtcvad classifier (controlled True/False
flags per frame) rather than real audio, since this tests the
SEGMENTATION logic itself, not webrtcvad's own real classification
accuracy -- that's exercised separately, with real audio, in
evaluate_pipeline_corpus.py/evaluate_real_recordings.py.
"""

from __future__ import annotations

from vad import FRAME_BYTES, FRAME_MS, VAD

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


class FakeVadEngine:
    """Returns a pre-set flag per frame index, called in order --
    lets a test control exactly which frames are "speech" without
    depending on real audio content or webrtcvad's own classifier."""

    def __init__(self, flags: list[bool]):
        self._flags = flags
        self._i = 0

    def is_speech(self, _frame: bytes, _sample_rate: int) -> bool:
        flag = self._flags[self._i]
        self._i += 1
        return flag


def _make_pcm(n_frames: int) -> bytes:
    return b"\x00" * (FRAME_BYTES * n_frames)  # content is irrelevant -- FakeVadEngine ignores it


print("=== Normal case: a speech run well under the cap behaves exactly as before ===")

vad = VAD(pre_roll_ms=0, post_roll_ms=0, max_segment_ms=8000)  # cap = ~266 frames at 30ms/frame
n_speech_frames = 20  # 600ms -- nowhere near the 8000ms cap
flags = [True] * n_speech_frames + [False] * 5
vad._vad = FakeVadEngine(flags)
segments = vad.segment_speech(_make_pcm(len(flags)))
# pre_roll_frames/post_roll_frames floor at 1 even when *_ms=0 (see
# VAD.__init__'s max(1, ...)), so the natural end (frame 20) gets exactly
# 1 frame of post-roll: (20+1)*FRAME_MS.
check(
    "a short, naturally-ending speech run produces exactly one segment",
    len(segments) == 1 and segments[0].end_ms == (n_speech_frames + 1) * FRAME_MS,
    [(s.start_ms, s.end_ms) for s in segments],
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== The real bug: continuous speech-classified audio with NO pause, longer than the cap ===")

vad2 = VAD(pre_roll_ms=0, post_roll_ms=0, max_segment_ms=3000)  # cap = 100 frames at 30ms/frame
n_total_frames = 250  # 7.5s of uninterrupted "speech" -- 2.5x the cap, real webrtcvad would do
flags2 = [True] * n_total_frames
vad2._vad = FakeVadEngine(flags2)
segments2 = vad2.segment_speech(_make_pcm(n_total_frames))
check(
    "a continuous run longer than the cap is split into multiple bounded segments, not one giant one",
    len(segments2) > 1,
    f"{len(segments2)} segment(s): {[(s.start_ms, s.end_ms) for s in segments2]}",
)
check(
    "the FIRST segment ends at (approximately) the cap, not at the end of the whole buffer",
    segments2[0].end_ms <= 3000 + FRAME_MS and segments2[0].end_ms < n_total_frames * FRAME_MS,
    (segments2[0].start_ms, segments2[0].end_ms),
)
check(
    "the first segment's end is strictly before the total buffered duration -- the real finalization proof api.py needs",
    segments2[0].end_ms < vad2.total_ms(_make_pcm(n_total_frames)),
    f"end_ms={segments2[0].end_ms} total_ms={vad2.total_ms(_make_pcm(n_total_frames))}",
)
check(
    "segments tile the whole continuous run with no gap (nothing silently dropped)",
    segments2[0].start_ms == 0 and segments2[-1].end_ms == n_total_frames * FRAME_MS,
    str([(s.start_ms, s.end_ms) for s in segments2]),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Pre-roll/post-roll still applied normally around forced cuts ===")

vad3 = VAD(pre_roll_ms=60, post_roll_ms=60, max_segment_ms=3000)  # pre/post = 2 frames each
flags3 = [True] * 250
vad3._vad = FakeVadEngine(flags3)
segments3 = vad3.segment_speech(_make_pcm(250))
check(
    "post-roll padding is still added at each forced cut (segment end extends past the raw cap)",
    segments3[0].end_ms > 3000,
    (segments3[0].start_ms, segments3[0].end_ms),
)
check(
    "pre-roll padding is still added at the start of the next segment after a forced cut",
    segments3[1].start_ms < segments3[0].end_ms,
    f"seg0 end={segments3[0].end_ms} seg1 start={segments3[1].start_ms}",
)

print("=== Natural pauses: the next segment must not reach back into the previous one ===")

# the lone "Please" bug: word, short dip, tail -- with default 1.5 s padding the second
# segment used to start inside the first, so the same word was transcribed twice
vad4 = VAD()  # defaults: pre/post roll 1500 ms
flags4 = [True] * 6 + [False] * 55 + [True] * 8 + [False] * 60
vad4._vad = FakeVadEngine(flags4)
segments4 = vad4.segment_speech(_make_pcm(len(flags4)))
check(
    "two bursts after a natural pause give non-overlapping segments",
    len(segments4) == 2 and segments4[1].start_ms >= segments4[0].end_ms,
    [(s.start_ms, s.end_ms) for s in segments4],
)
check(
    "no audio dropped between them (the second starts exactly where the first ended)",
    len(segments4) == 2 and segments4[1].start_ms == segments4[0].end_ms,
    [(s.start_ms, s.end_ms) for s in segments4],
)

print("=== A pause shorter than the post-roll keeps one utterance in one segment ===")

# "I don't ... want to go home now": split at the first silent frame, the negation and the
# UDK were matched in separate segments (2026-09-26 negation clips)
vad5 = VAD()  # post-roll 1500 ms = 50 frames
flags5 = [True] * 5 + [False] * 20 + [True] * 30 + [False] * 60
vad5._vad = FakeVadEngine(flags5)
segments5 = vad5.segment_speech(_make_pcm(len(flags5)))
check(
    "two bursts 600 ms apart give ONE segment covering both",
    len(segments5) == 1 and segments5[0].start_ms == 0 and segments5[0].end_ms == (55 + 50) * FRAME_MS,
    [(s.start_ms, s.end_ms) for s in segments5],
)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
