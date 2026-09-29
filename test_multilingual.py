"""
Multilingual support checks (English/Hindi/Telugu/Kannada/Tamil, added on
request -- see udks.py's module docstring for the full real-evidence
story: a real bug found in _normalize() that would have corrupted every
non-Latin-script phrase, and a real finding that the English semantic
model doesn't meaningfully work for Telugu/Kannada/Tamil at all).

Fast and portable -- exact-match/API-wiring checks only, no real STT/
audio. The real per-language semantic-matcher calibration (LaBSE, 200
negative pairs per language) and the _normalize() Unicode-category fix
were verified manually this session with real numbers (see README.md),
same treatment kws.py's/semantic.py's other real backends already got.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api
from udk_engine import _normalize, match_transcript
from udks import SUPPORTED_LANGUAGES, _UDK_TABLE, general_udks

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


print("=== _normalize() preserves Unicode combining marks (real bug, fixed) ===")

# "मुझे" (Hindi "mujhe", = "me") -- the old \w-based regex stripped both
# vowel signs, turning this into the different/meaningless "मझ".
check("Hindi combining marks survive normalization", _normalize("मुझे मदद चाहिए!") == "मुझे मदद चाहिए")
check("Telugu combining marks survive normalization", _normalize("నాకు సహాయం కావాలి!") == "నాకు సహాయం కావాలి")
check("Kannada combining marks survive normalization", _normalize("ನನಗೆ ಸಹಾಯ ಬೇಕು!") == "ನನಗೆ ಸಹಾಯ ಬೇಕು")
check("Tamil combining marks survive normalization", _normalize("எனக்கு உதவி வேண்டும்!") == "எனக்கு உதவி வேண்டும்")
check("English punctuation stripping unchanged", _normalize("Help me, please!") == "help me please")
check("English contractions unchanged (apostrophe still stripped)", _normalize("he's going to hurt me") == "hes going to hurt me")

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Exact-match layer works correctly for every supported language ===")

for lang in SUPPORTED_LANGUAGES:
    udks = general_udks(lang)
    all_ok = all(
        (result := match_transcript(udk.phrase, udks)).udk is not None
        and result.udk.udk_id == udk.udk_id
        and result.layer == "exact"
        and result.confidence == 1.0
        for udk in udks
    )
    check(f"all {len(udks)} UDKs exact-self-match correctly in {lang!r}", all_ok, lang)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Live API: no client-declared language -- journeys start pending detection ===")

client = TestClient(api.app, headers={"X-Service-Token": api.DEV_SERVICE_TOKEN})

resp = client.post("/v1/journeys/start", json={"user_id": "U_multilingual_test"})
check("a journey starts successfully with no language field at all (200)", resp.status_code == 200, str(resp.text))
if resp.status_code == 200:
    journey = api._store.get(resp.json()["journey_id"])
    check(
        "journey.language is None (pending real detection, not client-declared)",
        journey.language is None,
        journey.language,
    )
    check(
        "journey runs on English defaults until detection resolves it",
        all(u.language == "en" for u in journey.engine.udks if u.udk_type == "GENERAL"),
        str({u.udk_id: u.language for u in journey.engine.udks}),
    )

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Automatic language detection (language_id.py), fake detectors -- no real model load ===")


class FakeLanguageDetector:
    """Test double: returns a fixed (language, probability) regardless
    of the audio given, so the routing/hysteresis logic can be tested
    without loading faster-whisper or VoxLingua107."""

    def __init__(self, language: str, prob: float = 0.9):
        self.language = language
        self.prob = prob

    def detect(self, pcm: bytes) -> tuple[str, float]:
        return self.language, self.prob


def _fake_journey(**overrides) -> "api.JourneyState":
    from udks import SAMPLE_PERSONAL_UDK

    stt_by_language = {"hi": FakeSTT(), "te": FakeSTT()}
    kws_by_language = {}
    defaults = dict(
        journey_id="J_lid_test",
        user_id="U_lid_test",
        session_token="tok",
        engine=api.UDKEngine(api.all_udks(SAMPLE_PERSONAL_UDK, language="en")),
        stt=FakeSTT(),
        personal_udk=SAMPLE_PERSONAL_UDK,
        language=None,
        stt_by_language=stt_by_language,
        kws_by_language=kws_by_language,
    )
    defaults.update(overrides)
    return api.JourneyState(**defaults)


class FakeSTT:
    model_version = "fake-stt"

    def transcribe(self, pcm: bytes):
        from stt import Transcript

        return Transcript(text="", model_version=self.model_version)


six_seconds_silence = b"\x00\x00" * (6 * 16_000 + 1000)  # real byte-length math, no real audio needed -- total_ms() just counts bytes
short_silence = b"\x00\x00" * 8_000  # ~0.5s

journey = _fake_journey(language_detector=FakeLanguageDetector("te"))
journey.audio_buffer.extend(short_silence)
api._maybe_update_language(journey)
check(
    "fast detector resolves a brand-new (pending) journey's language on the first available audio",
    journey.language == "te" and journey.stt is journey.stt_by_language["te"],
    journey.language,
)
check(
    "resolving the language also rebuilds the engine with that language's UDKs",
    all(u.language == "te" for u in journey.engine.udks if u.udk_type == "GENERAL"),
    str({u.udk_id: u.language for u in journey.engine.udks}),
)

hindi_journey = _fake_journey(language_detector=FakeLanguageDetector("hi"))
hindi_journey.audio_buffer.extend(short_silence)
api._maybe_update_language(hindi_journey)
hindi_udk08 = _UDK_TABLE["hi"]["UDK_08"][0]
decision = hindi_journey.engine.decide(hindi_udk08, now_s=0.0)
check(
    "after real audio-based detection resolves Hindi, a real Hindi UDK phrase (UDK_08 equivalent) still triggers correctly",
    decision.udk is not None and decision.udk.udk_id == "UDK_08" and decision.decision != "NO_ACTION",
    str(decision),
)

journey_no_detector = _fake_journey(language_detector=None)
journey_no_detector.audio_buffer.extend(short_silence)
api._maybe_update_language(journey_no_detector)
check(
    "with no detector configured (feature not enabled), a pending journey falls back to English rather than staying undetected forever",
    journey_no_detector.language == "en",
    journey_no_detector.language,
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print(f"=== Periodic recheck: {api.LID_SWITCH_CONFIRMATIONS} consecutive agreeing detections required before switching ===")
print("(real-world audio testing found 2 wasn't enough -- see LID_SWITCH_CONFIRMATIONS's docstring in api.py:")
print(" real background music/noise made VoxLingua107 confidently wrong for 2 consecutive real 6s windows,")
print(" which used to wrongly flip a real all-English clip's journey onto the Kannada STT backend mid-stream)")

journey2 = _fake_journey(language="en", language_recheck_detector=FakeLanguageDetector("hi"))
journey2.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey2)
check(
    "a single disagreeing recheck does NOT switch the language yet (avoids flip-flopping on one noisy signal)",
    journey2.language == "en",
    journey2.language,
)
check("the disagreement is recorded as a pending candidate", journey2._lid_pending_candidate == "hi", journey2._lid_pending_candidate)

journey2.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey2)
check(
    "a SECOND consecutive agreeing recheck alone no longer switches -- this is the real bug fix (was: switched here)",
    journey2.language == "en" and journey2._lid_pending_streak == 2,
    (journey2.language, journey2._lid_pending_streak),
)

journey2.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey2)
check(
    f"a THIRD consecutive agreeing recheck DOES switch (real signal sustained across {api.LID_SWITCH_CONFIRMATIONS} independent windows, not a 2-window noise burst)",
    journey2.language == "hi" and journey2.stt is journey2.stt_by_language["hi"],
    journey2.language,
)
check("the pending candidate/streak reset after a confirmed switch", journey2._lid_pending_candidate is None and journey2._lid_pending_streak == 0)

journey3 = _fake_journey(language="en", language_recheck_detector=FakeLanguageDetector("hi"))
journey3.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey3)
journey3.language_recheck_detector = FakeLanguageDetector("en")  # back to agreeing with the active language
journey3.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey3)
check(
    "a disagreement NOT confirmed by the next recheck resets cleanly, no spurious switch",
    journey3.language == "en" and journey3._lid_pending_candidate is None and journey3._lid_pending_streak == 0,
    (journey3.language, journey3._lid_pending_candidate, journey3._lid_pending_streak),
)

journey4 = _fake_journey(language="en", language_recheck_detector=FakeLanguageDetector("kn"))
journey4.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey4)  # kn, streak=1
journey4.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey4)  # kn, streak=2 -- this is exactly where the real bug used to fire
journey4.language_recheck_detector = FakeLanguageDetector("en")  # real content resumes agreeing with English
journey4.audio_buffer.extend(six_seconds_silence)
api._maybe_update_language(journey4)
check(
    "real-world regression: 2 consecutive wrong agreements followed by a correct one never switches at all (matches the real negatives_6 clip's actual score sequence)",
    journey4.language == "en" and journey4._lid_pending_candidate is None and journey4._lid_pending_streak == 0,
    (journey4.language, journey4._lid_pending_candidate, journey4._lid_pending_streak),
)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== create_or_get selects the right STT backend per language (fake backends, no real model load) ===")


class FakeTeluguSTT:
    model_version = "fake-telugu-stt"

    def transcribe(self, pcm):
        raise NotImplementedError  # never actually called in this check


fake_te_stt = FakeTeluguSTT()
default_stt = api.MockSTT()
telugu_journey, _ = api._store.create_or_get(
    "J_telugu_stt_test", "U_telugu_stt_test", default_stt, api.SAMPLE_PERSONAL_UDK, language="te", stt_by_language={"te": fake_te_stt}
)
check("a language with a real entry in stt_by_language gets that backend, not the default", telugu_journey.stt is fake_te_stt)

kannada_journey, _ = api._store.create_or_get(
    "J_kannada_stt_test", "U_kannada_stt_test", default_stt, api.SAMPLE_PERSONAL_UDK, language="kn", stt_by_language={"te": fake_te_stt}
)
check("a language with NO entry in stt_by_language falls back to the default stt backend", kannada_journey.stt is default_stt)

no_override_journey, _ = api._store.create_or_get(
    "J_no_override_stt_test", "U_no_override_stt_test", default_stt, api.SAMPLE_PERSONAL_UDK, language="te"
)
check("stt_by_language omitted entirely (backward compat) -- default constructor value is used, not an error", no_override_journey.stt is default_stt)

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== create_or_get selects the right KWS backend per language (fake backends, no real model load) ===")


class FakeTeluguKWS:
    model_version = "fake-telugu-kws"

    def enroll(self, phrase_id, pcm):
        raise NotImplementedError

    def spot(self, pcm):
        raise NotImplementedError

    def clear(self, phrase_id):
        raise NotImplementedError


fake_te_kws = FakeTeluguKWS()
telugu_kws_journey, _ = api._store.create_or_get(
    "J_telugu_kws_test", "U_telugu_kws_test", default_stt, api.SAMPLE_PERSONAL_UDK, language="te", kws_by_language={"te": fake_te_kws}
)
check("a language with a real entry in kws_by_language gets that backend, not the default (None)", telugu_kws_journey.kws is fake_te_kws)

kannada_kws_journey, _ = api._store.create_or_get(
    "J_kannada_kws_test", "U_kannada_kws_test", default_stt, api.SAMPLE_PERSONAL_UDK, language="kn", kws_by_language={"te": fake_te_kws}
)
check("a language with NO entry in kws_by_language falls back to the default (None -- KWS off)", kannada_kws_journey.kws is None)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
