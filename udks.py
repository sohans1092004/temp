"""
The 20 general UDK phrases (Section 16 of the review doc) plus a sample
personal UDK, with the TRIGGER_ALL / TRIGGER_VERIFY tier each phrase
defaults to per the doc's phrase-selection reasoning.

Two phrases were revised from Section 16's original list, using real
false-positive evidence gathered this session, not guesswork -- see
README.md for the measurements:

  - UDK_01 was "Call the police". A real live test found this exact
    phrase, embedded as a literal substring inside an ordinary teasing
    remark ("I will call the police on you"), fired the EXACT-match
    layer at confidence 1.0 -- the highest-trust layer, the one that
    skips straight to TRIGGER_ALL with no confirmation step. Short,
    generic phrases are especially exposed to this failure mode, since
    they're likely to appear as a literal substring of an unrelated
    sentence. Lengthened to "Call the police, I need help" -- no longer
    a literal substring of the joke phrase (so that case now lands at a
    confirmable TRIGGER_VERIFY instead of an unverified TRIGGER_ALL, the
    realistic ceiling for a lexical-similarity layer, not a claim of
    eliminating the ambiguity). This specific replacement was picked by
    MEASURING candidate phrases against the real ASR-noise regression
    test ("call the plice right now") first -- an earlier candidate,
    "Somebody call the police, please", solved the substring problem but
    measurably broke that ASR-noise case (its own fuzzy score dropped
    low enough that a DIFFERENT, unrelated UDK won instead). This phrase
    keeps the original's strong fuzzy-match robustness (0.837 raw
    partial_ratio vs. the ASR-noise transcript, close to the original's
    0.966) while no longer being a clean substring of the joke phrase.
  - UDK_20 ("I need to get out of here") measurably false-positived
    against ordinary conversation ("Let's get out of here", a plain
    exit-a-party remark) in evaluate_pipeline_corpus.py's real corpus.
    It has the same character as UDK_14/UDK_18 below -- natural in both
    a real emergency AND an ordinary tense-but-safe moment -- so it gets
    the same verify-by-default treatment rather than a full-alarm
    default: not a claim that it doesn't matter, just that a single
    unconfirmed occurrence shouldn't skip the confirmation step, exactly
    like the other two already don't.

This is a documented example of the "narrow the phrase, not eliminate
the tension" tradeoff discussed in this session: Section 16's own
selection criteria already require phrases to be natural under distress
AND uncommon in ordinary speech -- those two constraints are in real
tension, and this is a small, evidence-driven step along that spectrum,
not a claim the tension is resolved. The safety team still owns the
final list (Section 16).

MULTILINGUAL SUPPORT (added on request -- India, at least English/Hindi/
Telugu/Kannada/Tamil): each general UDK now has a translated phrase per
supported language, keyed by udk_id so the SAME logical phrase (e.g.
"the call-the-police-and-ask-for-help one") exists in each language,
rather than treating each language as an unrelated list. Real,
foundational findings from building this, not assumptions:

  - `udk_engine.py`'s `_normalize()` had a REAL BUG that would have
    silently broken every non-Latin-script phrase: its regex relied on
    Python's `\\w`, which does not include Unicode COMBINING MARKS
    (category Mn/Mc) -- the vowel signs (matras) that carry essential
    meaning in Devanagari/Telugu/Kannada/Tamil. It turned "मुझे" (Hindi
    "mujhe", = "me") into "मझ" (garbled, a different string), which
    would have corrupted exact/fuzzy matching for every phrase in these
    languages. Fixed by stripping actual Unicode punctuation/symbol
    categories (P*/S*) instead of relying on \\w's incomplete
    letter-vs-mark distinction -- verified to strip English punctuation
    identically to the old behavior while preserving Indic combining
    marks (see README.md for the before/after).
  - Translations below are a first pass (produced directly, not sourced
    from a professional translator or native-speaker safety reviewer)
    -- exactly the kind of judgment call Section 16 already says the
    safety team owns for the English list; more true here, for phrases
    someone may need to say under real distress in their first language.
    Treat these as a working starting point to validate the pipeline
    end to end, not a finished, reviewed phrase list.
  - `verify_by_default` flags are carried over 1:1 from the English
    assignment (the UDK_14/UDK_18/UDK_20-equivalent phrases) as a
    starting assumption, NOT independently re-measured against a real
    per-language false-positive corpus the way the English list's flags
    were (see the two revisions documented above, both found by testing,
    not guessed). Whether "मुझे यहां सुरक्षित महसूस नहीं हो रहा" is
    similarly common in ordinary Hindi conversation hasn't been checked
    the same way.
  - KWS (`kws.py`, `separation.py`) stays English-only for now (its
    wav2vec2 model is trained on English audio) -- this was an explicit
    scope decision, not an oversight: making it multilingual means
    swapping to a multilingual wav2vec2 model (e.g. XLSR-53) AND
    separately re-calibrating its match threshold per language against a
    real corpus, a materially bigger effort than STT/matching, deferred
    rather than done partially.
"""

from dataclasses import dataclass
from typing import Literal

UDKType = Literal["GENERAL", "PERSONAL"]
Language = Literal["en", "hi", "te", "kn", "ta"]


@dataclass(frozen=True)
class UDK:
    udk_id: str
    phrase: str
    udk_type: UDKType
    verify_by_default: bool = False
    language: Language = "en"


# udk_id -> (phrase, verify_by_default), one row per language. Keeping a
# single source of table keyed by language avoids 20 UDK objects x 5
# languages worth of near-duplicate lines and keeps each language's
# ordering/verify-by-default flags visibly parallel to the English list.
_UDK_TABLE: dict[str, dict[str, tuple[str, bool]]] = {
    "en": {
        "UDK_01": ("Call the police, I need help", False),
        "UDK_02": ("I'm not safe", False),
        "UDK_03": ("Get away from me", False),
        "UDK_04": ("Someone is following me", False),
        "UDK_05": ("I need help right now", False),
        "UDK_06": ("Don't touch me", False),
        "UDK_07": ("I'm being followed", False),
        "UDK_08": ("Help me, please", False),
        "UDK_09": ("I'm scared, stay back", False),
        "UDK_10": ("Let go of me", False),
        "UDK_11": ("I need the police here", False),
        "UDK_12": ("Someone is trying to hurt me", False),
        "UDK_13": ("Please don't hurt me", False),
        "UDK_14": ("I don't feel safe here", True),
        "UDK_15": ("Stay away from me", False),
        "UDK_16": ("I'm in danger", False),
        "UDK_17": ("Somebody help me now", False),
        "UDK_18": ("I want to go home now", True),
        "UDK_19": ("Please call for help", False),
        "UDK_20": ("I need to get out of here", True),
    },
    "hi": {
        "UDK_01": ("पुलिस को बुलाओ, मुझे मदद चाहिए", False),
        "UDK_02": ("मैं सुरक्षित नहीं हूं", False),
        "UDK_03": ("मुझसे दूर हटो", False),
        "UDK_04": ("कोई मेरा पीछा कर रहा है", False),
        "UDK_05": ("मुझे अभी मदद चाहिए", False),
        "UDK_06": ("मुझे मत छुओ", False),
        "UDK_07": ("मेरा पीछा किया जा रहा है", False),
        "UDK_08": ("मदद करो, कृपया", False),
        "UDK_09": ("मुझे डर लग रहा है, दूर रहो", False),
        "UDK_10": ("मुझे छोड़ दो", False),
        "UDK_11": ("मुझे यहां पुलिस चाहिए", False),
        "UDK_12": ("कोई मुझे नुकसान पहुंचाने की कोशिश कर रहा है", False),
        "UDK_13": ("कृपया मुझे मत मारो", False),
        "UDK_14": ("मुझे यहां सुरक्षित महसूस नहीं हो रहा", True),
        "UDK_15": ("मुझसे दूर रहो", False),
        "UDK_16": ("मैं खतरे में हूं", False),
        "UDK_17": ("कोई अभी मेरी मदद करो", False),
        "UDK_18": ("मुझे अभी घर जाना है", True),
        "UDK_19": ("कृपया मदद के लिए कॉल करो", False),
        "UDK_20": ("मुझे यहां से निकलना है", True),
    },
    "te": {
        "UDK_01": ("పోలీసులకు కాల్ చేయండి, నాకు సహాయం కావాలి", False),
        "UDK_02": ("నేను సురక్షితంగా లేను", False),
        "UDK_03": ("నా నుండి దూరంగా వెళ్ళు", False),
        "UDK_04": ("ఎవరో నన్ను వెంబడిస్తున్నారు", False),
        "UDK_05": ("నాకు ఇప్పుడే సహాయం కావాలి", False),
        "UDK_06": ("నన్ను తాకవద్దు", False),
        "UDK_07": ("నన్ను ఎవరో వెంబడిస్తున్నారు", False),
        "UDK_08": ("దయచేసి నాకు సహాయం చేయండి", False),
        "UDK_09": ("నాకు భయంగా ఉంది, దూరంగా ఉండు", False),
        "UDK_10": ("నన్ను వదిలేయ్", False),
        "UDK_11": ("నాకు ఇక్కడ పోలీసులు కావాలి", False),
        "UDK_12": ("ఎవరో నన్ను గాయపరచడానికి ప్రయత్నిస్తున్నారు", False),
        "UDK_13": ("దయచేసి నన్ను గాయపరచవద్దు", False),
        "UDK_14": ("నాకు ఇక్కడ సురక్షితంగా అనిపించడం లేదు", True),
        "UDK_15": ("నా నుండి దూరంగా ఉండు", False),
        "UDK_16": ("నేను ప్రమాదంలో ఉన్నాను", False),
        "UDK_17": ("ఎవరైనా ఇప్పుడే నాకు సహాయం చేయండి", False),
        "UDK_18": ("నాకు ఇప్పుడే ఇంటికి వెళ్లాలి", True),
        "UDK_19": ("దయచేసి సహాయం కోసం కాల్ చేయండి", False),
        "UDK_20": ("నేను ఇక్కడ నుండి బయటపడాలి", True),
    },
    "kn": {
        "UDK_01": ("ಪೊಲೀಸರಿಗೆ ಕರೆ ಮಾಡಿ, ನನಗೆ ಸಹಾಯ ಬೇಕು", False),
        "UDK_02": ("ನಾನು ಸುರಕ್ಷಿತವಾಗಿಲ್ಲ", False),
        "UDK_03": ("ನನ್ನಿಂದ ದೂರ ಹೋಗು", False),
        "UDK_04": ("ಯಾರೋ ನನ್ನನ್ನು ಹಿಂಬಾಲಿಸುತ್ತಿದ್ದಾರೆ", False),
        "UDK_05": ("ನನಗೆ ಈಗಲೇ ಸಹಾಯ ಬೇಕು", False),
        "UDK_06": ("ನನ್ನನ್ನು ಮುಟ್ಟಬೇಡ", False),
        "UDK_07": ("ನನ್ನನ್ನು ಯಾರೋ ಹಿಂಬಾಲಿಸುತ್ತಿದ್ದಾರೆ", False),
        "UDK_08": ("ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ", False),
        "UDK_09": ("ನನಗೆ ಭಯವಾಗುತ್ತಿದೆ, ದೂರವಿರು", False),
        "UDK_10": ("ನನ್ನನ್ನು ಬಿಡು", False),
        "UDK_11": ("ನನಗೆ ಇಲ್ಲಿ ಪೊಲೀಸರು ಬೇಕು", False),
        "UDK_12": ("ಯಾರೋ ನನಗೆ ಹಾನಿ ಮಾಡಲು ಪ್ರಯತ್ನಿಸುತ್ತಿದ್ದಾರೆ", False),
        "UDK_13": ("ದಯವಿಟ್ಟು ನನಗೆ ಹಾನಿ ಮಾಡಬೇಡಿ", False),
        "UDK_14": ("ನನಗೆ ಇಲ್ಲಿ ಸುರಕ್ಷಿತ ಎನಿಸುತ್ತಿಲ್ಲ", True),
        "UDK_15": ("ನನ್ನಿಂದ ದೂರವಿರು", False),
        "UDK_16": ("ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ", False),
        "UDK_17": ("ಯಾರಾದರೂ ಈಗಲೇ ನನಗೆ ಸಹಾಯ ಮಾಡಿ", False),
        "UDK_18": ("ನನಗೆ ಈಗಲೇ ಮನೆಗೆ ಹೋಗಬೇಕು", True),
        "UDK_19": ("ದಯವಿಟ್ಟು ಸಹಾಯಕ್ಕಾಗಿ ಕರೆ ಮಾಡಿ", False),
        "UDK_20": ("ನಾನು ಇಲ್ಲಿಂದ ಹೊರಬರಬೇಕು", True),
    },
    "ta": {
        "UDK_01": ("போலீசை அழையுங்கள், எனக்கு உதவி வேண்டும்", False),
        "UDK_02": ("நான் பாதுகாப்பாக இல்லை", False),
        "UDK_03": ("என்னிடமிருந்து விலகிச் செல்", False),
        "UDK_04": ("யாரோ என்னைப் பின்தொடர்கிறார்கள்", False),
        "UDK_05": ("எனக்கு இப்போதே உதவி தேவை", False),
        "UDK_06": ("என்னைத் தொடாதே", False),
        "UDK_07": ("என்னை யாரோ பின்தொடர்கிறார்கள்", False),
        "UDK_08": ("தயவுசெய்து எனக்கு உதவுங்கள்", False),
        "UDK_09": ("எனக்குப் பயமாக இருக்கிறது, தூரமாக இரு", False),
        "UDK_10": ("என்னை விடு", False),
        "UDK_11": ("எனக்கு இங்கே போலீஸ் வேண்டும்", False),
        "UDK_12": ("யாரோ என்னைக் காயப்படுத்த முயற்சிக்கிறார்கள்", False),
        "UDK_13": ("தயவுசெய்து என்னைத் துன்புறுத்தாதே", False),
        "UDK_14": ("இங்கே எனக்குப் பாதுகாப்பாக உணரவில்லை", True),
        "UDK_15": ("என்னிடமிருந்து விலகி இரு", False),
        "UDK_16": ("நான் ஆபத்தில் இருக்கிறேன்", False),
        "UDK_17": ("யாராவது இப்போதே எனக்கு உதவுங்கள்", False),
        "UDK_18": ("எனக்கு இப்போதே வீட்டிற்குப் போக வேண்டும்", True),
        "UDK_19": ("தயவுசெய்து உதவிக்கு அழையுங்கள்", False),
        "UDK_20": ("நான் இங்கிருந்து வெளியேற வேண்டும்", True),
    },
}

SUPPORTED_LANGUAGES: tuple[Language, ...] = ("en", "hi", "te", "kn", "ta")

GENERAL_UDKS_BY_LANGUAGE: dict[str, list[UDK]] = {
    lang: [UDK(udk_id, phrase, "GENERAL", verify_by_default=verify, language=lang) for udk_id, (phrase, verify) in table.items()]
    for lang, table in _UDK_TABLE.items()
}

# Backward-compat alias -- every existing caller (api.py, pipeline.py,
# evaluate_pipeline_corpus.py, the whole test suite) imports this name
# expecting the English list; keeping it means none of them need to
# change just because other languages exist now.
GENERAL_UDKS: list[UDK] = GENERAL_UDKS_BY_LANGUAGE["en"]

# One sample personal UDK, standing in for a per-user enrolled phrase
# (Section 5: stored as text, speaker-independent detection).
SAMPLE_PERSONAL_UDK = UDK("PUDK_U1001", "the weather in Denver is lovely", "PERSONAL")


def general_udks(language: Language = "en") -> list[UDK]:
    return GENERAL_UDKS_BY_LANGUAGE[language]


def all_udks(personal: UDK = SAMPLE_PERSONAL_UDK, language: Language = "en") -> list[UDK]:
    return general_udks(language) + [personal]
