"""
Voice Activity Detection wrapper around webrtcvad.

Per Section 2 of the review: VAD should bias toward treating ambiguous
audio as speech, not silence, so a low aggressiveness setting is the
default. Frames are 16-bit mono PCM at 8/16/32/48kHz, 10/20/30ms each,
which is what webrtcvad requires.
"""

from __future__ import annotations

import webrtcvad
from dataclasses import dataclass

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_MS / 1000.0)) * 2  # 16-bit samples


@dataclass
class SpeechSegment:
    start_ms: int
    end_ms: int
    pcm: bytes


class VAD:
    def __init__(
        self,
        aggressiveness: int = 1,
        pre_roll_ms: int = 1500,
        post_roll_ms: int = 1500,
        max_segment_ms: int = 8000,
    ):
        """
        aggressiveness: 0 (most permissive / most likely to call things
        speech) to 3 (most aggressive at rejecting non-speech). Section 2
        recommends biasing toward speech, so this defaults low.

        max_segment_ms: real bug found and fixed testing against real
        continuous audio (background music, sound effects, non-stop
        talk) -- a genuinely important condition for a personal-safety
        product this project hadn't tested against before. Without a
        cap, a continuous run of speech-classified frames never hits a
        non-speech frame, so the segment keeps growing to match whatever
        has arrived so far -- `end` always equals the current buffer
        length, so api.py's finalization check (`seg.end_ms < total`,
        "more audio arrived after this segment ended, proving it's
        done") can never become true. In a real test (a 56-second real
        video clip VAD classified as one unbroken speech run), this
        meant the ENTIRE clip stayed one open segment and was never
        transcribed or matched until the journey was explicitly stopped
        -- silently falling back to "wait for everything," exactly what
        this system is supposed to never do. 8s is a real, deliberate
        starting choice (long enough for a realistic UDK-length
        utterance even under a noisy floor, short enough to bound
        worst-case detection latency), not yet swept against real data
        the way e.g. kws.py's threshold was -- a real next step, not a
        finished calibration.
        """
        self._vad = webrtcvad.Vad(aggressiveness)
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.post_roll_frames = max(1, post_roll_ms // FRAME_MS)
        self.max_segment_frames = max(1, max_segment_ms // FRAME_MS)

    def _frames(self, pcm: bytes):
        for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
            yield pcm[i : i + FRAME_BYTES]

    def total_ms(self, pcm: bytes) -> int:
        """Duration accounted for by whole frames in this buffer, in the
        same units as SpeechSegment.end_ms. Callers use this to tell
        whether a segment has "cleared" the buffer's tail -- i.e. more
        audio has arrived since it ended, proving it's done growing --
        without needing to know FRAME_MS/FRAME_BYTES themselves."""
        return (len(pcm) // FRAME_BYTES) * FRAME_MS

    def segment_speech(self, pcm: bytes) -> list[SpeechSegment]:
        """
        Split raw 16kHz mono PCM into speech segments, each padded with
        pre-roll/post-roll (Section 2: don't cut exactly at the VAD
        boundary, or a detection loses the context around it).
        """
        frames = list(self._frames(pcm))
        flags = [self._vad.is_speech(f, SAMPLE_RATE) for f in frames]

        segments: list[SpeechSegment] = []
        i = 0
        n = len(flags)
        prev_end, prev_cut = 0, False
        while i < n:
            if flags[i]:
                start = max(0, i - self.pre_roll_frames)
                # Real bug (2026-09-25, a lone "Please" recording): a second burst within
                # the 1.5 s pre-roll reached back over the previous segment, so ONE word was
                # transcribed twice -> counted as a repeated UDK -> escalated to TRIGGER_ALL.
                # Overlap is kept only after a forced max-length cut (see below), where it
                # stops a phrase split at the cut from being lost.
                if not prev_cut:
                    start = max(start, prev_end)
                j = i
                # Capped at max_segment_frames -- see __init__'s
                # max_segment_ms docstring for the real bug this closes:
                # without the cap, a continuous run of speech-classified
                # frames (real background music/noise/non-stop talk) never
                # hits a non-speech frame to stop at, so `j` -- and thus
                # `end` -- would just keep growing to match the buffer,
                # and this segment would never satisfy the caller's
                # "more audio arrived after this ended" finalization
                # proof. Cutting here forcibly closes it instead, so it
                # gets transcribed/matched on a bounded schedule; the
                # pre-roll of whatever segment starts next still overlaps
                # back into this one's post-roll, so a phrase split
                # across the cut isn't silently dropped from both sides.
                run_limit = i + self.max_segment_frames
                # A pause shorter than the post-roll does not end the segment (2026-09-26):
                # stopping at the first silent frame split "I don't | want to go home now"
                # into two segments once the pre-roll no longer overlapped (above), so the
                # negation and the UDK were matched apart. One utterance = one segment.
                last = i  # last speech frame of this segment
                while j < n and j < run_limit and j - last <= self.post_roll_frames:
                    if flags[j]:
                        last = j
                    j += 1
                end = min(n, last + 1 + self.post_roll_frames)
                prev_end = end
                prev_cut = j >= run_limit and j < n and j - last <= self.post_roll_frames  # cut mid-speech by the cap
                seg_pcm = b"".join(frames[start:end])
                segments.append(
                    SpeechSegment(
                        start_ms=start * FRAME_MS,
                        end_ms=end * FRAME_MS,
                        pcm=seg_pcm,
                    )
                )
                i = end
            else:
                i += 1
        return segments
