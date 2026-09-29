"""
Personal UDK enrollment (Section 5): phrase-based, speaker-independent by
default. The phrase text is the primary and only long-term artifact --
any enrollment audio is used only in memory here, to sanity-check the
phrase, and is never itself the thing this module persists.

Two checks, matching what Section 5 says enrollment-time verification is
actually good for:
  - distinctiveness: is the phrase different enough from ordinary
    conversation that it won't misfire constantly? Not a biometric gate.
  - intelligibility: can STT/matching pick it up at all, from one sample?
    A sanity check on the phrase choice -- not a claim that this exact
    enrollment recording predicts real-emergency reliability (Section 5
    is explicit that calm, clean enrollment audio doesn't predict a
    stressed, shouted, out-of-breath run-time utterance).
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from kws import KWSBackend
from stt import STTBackend
from udk_engine import _normalize
from udks import UDK

# Stand-in for "ordinary conversation" -- enough to catch an obviously
# too-generic phrase choice without needing a real corpus for what's
# meant to be a lightweight sanity check, not a production classifier.
_ORDINARY_CONVERSATION = [
    "how are you doing",
    "what a lovely day",
    "I need to go now",
    "can you help me with this",
    "I'm really tired",
    "call me later",
    "let's get out of here",
    "I'm on my way",
    "see you soon",
]

MIN_WORDS = 3
DISTINCTIVENESS_MAX = 0.75  # at/above this, too close to ordinary conversation
INTELLIGIBILITY_MIN = 0.6  # same bar as udk_engine's TRIGGER_VERIFY threshold


@dataclass
class EnrollmentResult:
    udk: UDK | None
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.udk is not None


def _distinctiveness_errors(phrase: str) -> list[str]:
    errors = []
    words = _normalize(phrase).split()
    if len(words) < MIN_WORDS:
        errors.append(f"phrase is too short ({len(words)} word(s)) -- pick something more distinctive")
    for common in _ORDINARY_CONVERSATION:
        score = fuzz.token_set_ratio(_normalize(phrase), _normalize(common)) / 100.0
        if score >= DISTINCTIVENESS_MAX:
            errors.append(f'phrase is too close to ordinary conversation ("{common}") -- pick something less common')
            break
    return errors


def _intelligibility_errors(phrase: str, sample_pcm: bytes, stt: STTBackend) -> list[str]:
    transcript = stt.transcribe(sample_pcm)
    score = fuzz.token_set_ratio(_normalize(phrase), _normalize(transcript.text)) / 100.0
    if score < INTELLIGIBILITY_MIN:
        return [
            f"sample audio didn't clearly match the phrase (transcribed as {transcript.text!r}) "
            "-- try recording again somewhere quieter"
        ]
    return []


def enroll(
    user_id: str,
    phrase: str,
    sample_pcm: bytes | None = None,
    stt: STTBackend | None = None,
    kws: KWSBackend | None = None,
) -> EnrollmentResult:
    """Validate and produce a personal UDK. If sample_pcm/stt are given,
    the audio is used only for the intelligibility check above -- the
    caller must not persist it (Section 5's "store phrase as text, audio
    only transiently" rule applies at the call site, not inside this
    function, since this function never writes anything to disk itself).

    If kws is also given, that same sample_pcm becomes the personal UDK's
    KWS reference clip (Section 4's Approach E extended to personal UDKs,
    not just the 20 general ones) -- real user audio, not TTS, which is
    the better reference clip anyway when it's actually available.
    Nothing to enroll into if sample_pcm wasn't provided; that's fine,
    the personal UDK still works via STT-only matching same as before.

    Re-enrollment/versioning/deactivating the old phrase is the caller's
    concern (whatever holds the active personal UDK per user) -- this is
    a pure validate-and-construct step, not a store.
    """
    phrase = phrase.strip()
    errors = [] if phrase else ["phrase is empty"]
    if phrase:
        errors.extend(_distinctiveness_errors(phrase))
    if sample_pcm is not None and stt is not None and phrase:
        errors.extend(_intelligibility_errors(phrase, sample_pcm, stt))

    if errors:
        return EnrollmentResult(udk=None, errors=errors)

    udk = UDK(udk_id=f"PUDK_{user_id}", phrase=phrase, udk_type="PERSONAL")
    if sample_pcm is not None and kws is not None:
        kws.enroll(udk.udk_id, sample_pcm)
    return EnrollmentResult(udk=udk, errors=[])
