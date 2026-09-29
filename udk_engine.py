"""
UDK matching + the Section 6 decision contract: audio-derived signals in,
exactly one of TRIGGER_ALL / TRIGGER_VERIFY / NO_ACTION out.

Matching has three layers (Section 4):
  1. exact    - normalized substring/equality match
  2. fuzzy    - rapidfuzz, catches ASR noise/typos/word-order
  3. semantic - a real pretrained multilingual sentence-embedding model
                (semantic.py's SentenceTransformerSemanticMatcher,
                Section 17: no training needed, just inference) for
                paraphrase cases like "he's going to hurt me" matching
                "someone is threatening me" -- optional, passed in as
                match_transcript's/UDKEngine's semantic_matcher param.
                Falls back to rapidfuzz's token_set_ratio as a lighter
                stand-in when none is given, same as before this was
                built -- good enough to prove the pipeline shape, not a
                claim of real semantic accuracy on its own.

Thresholds are the Section 6 starting points, explicitly meant to be
tuned against the real test corpus later, not treated as final:
  confidence >= 0.85              -> TRIGGER_ALL
  0.6 <= confidence < 0.85        -> TRIGGER_VERIFY
  confidence < 0.6                -> NO_ACTION
Repetition of the same/related UDK within REPEAT_WINDOW_S can lift a
TRIGGER_VERIFY straight to TRIGGER_ALL, per Section 6/7's reasoning that
repetition is a confidence booster, never a requirement to fire at all.

Section 4's Approach E (KWS running in parallel with STT, revisited after
v1's deferral -- see kws.py and README.md for how): decide() takes an
optional kws_match alongside the transcript. The two branches are fused,
not just compared -- if they independently agree on the same UDK, that's
cross-validation from two unrelated detection mechanisms (an ASR+text
matcher vs. an audio-embedding+DTW matcher), treated the same way
repetition already is: strong enough evidence to justify TRIGGER_ALL on
its own, including overriding a verify-by-default phrase. If STT fails
outright (empty transcript, an outage per Section 3) but KWS still
matches, KWS alone can still produce a decision -- this is the actual
single-point-of-failure removal Section 4 was about, not just an accuracy
bump.

A third, optional signal (beats_distress_detector.py): a real trained
AudioSet classifier (Microsoft's BEATs, fine-tuned), scoring the max
probability across a curated set of vocal-distress classes (Screaming,
Yell, Shout, Crying/sobbing, etc). Superseded scream_detector.py's
pure-DSP heuristic (envelope roughness/pitch elevation/energy) after real
measurement found it a clear improvement: on this project's own real
corpus, the one confirmed real scream peaks at 0.454 vocal score in its
actual scream window vs. the DSP detector's 0.320 on the same clip, while
6 real negative clips maxed at 0.018 (vs. the DSP detector's 9.0% FPR).
Still corroboration-only, same as before -- unlike KWS it is NEVER
treated as an independent detection branch -- decide() only even looks at
scream_score after STT/KWS has already produced a TRIGGER_VERIFY-or-better
match on its own, where it can then add corroboration the same way
STT+KWS agreement already does. It can never turn a NO_ACTION into
anything else. scream_detector.py is left in place, standalone and still
independently testable, but is no longer wired into api.py.

Two more corroborating signals (stt_confidence_gate.py, dual_asr_guard.py),
added after a real investigation into overlapping-speech/degraded-audio
failures found free STT transcription can produce fully fabricated,
fluent text with no visible sign anything is wrong. Unlike scream_score,
these only ever DISCOUNT a text match's confidence, never add to it --
the "no single signal fires alone" principle applied in the other
direction. stt_suspect (free -- reuses faster-whisper's own decoding
metadata) catches genuinely-hallucinating output (the decoder's own
temperature fallback fired). dual_asr_disagreement (a second model's
independent transcript, run selectively -- see api.py's wiring) catches
"confidently wrong" output stt_suspect structurally cannot: a transcript
the decoder itself shows no sign of doubting, that is nevertheless not
what was actually said. Both are skipped when KWS independently
corroborates the match, since KWS never depends on the STT transcript's
text quality at all.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz import fuzz

from beats_distress_detector import DISTRESS_SCORE_THRESHOLD as SCREAM_SCORE_THRESHOLD
from kws import KWSMatch
from semantic import SemanticBackend
from udks import UDK

Decision = Literal["TRIGGER_ALL", "TRIGGER_VERIFY", "NO_ACTION"]

TRIGGER_ALL_THRESHOLD = 0.85
TRIGGER_VERIFY_THRESHOLD = 0.60
REPEAT_WINDOW_S = 15.0
# Multiple DIFFERENT UDKs firing within the window, not just the same one
# repeating -- e.g. "get away from me" then "call the police" -- is
# arguably a stronger signal than repetition: harder to explain away as
# one ambiguous phrase match. 2 is the "on top of the first" count, same
# shape as repetition's "this one plus at least one earlier."
DISTINCT_UDK_ESCALATION_COUNT = 2
# Starting point, not calibrated against Section 12's real corpus -- kws.py's
# spot() already thresholds distance into a binary match/no-match, so this
# is a fixed confidence for "KWS matched," landing in TRIGGER_VERIFY range
# on its own (independent-branch agreement is what pushes it to TRIGGER_ALL).
KWS_MATCH_CONFIDENCE = 0.65

# Real bug found and fixed: a SHORT transcript that's merely a substring
# of a longer UDK phrase was previously always scored confidence=1.0 (the
# exact layer's substring check has no length/uniqueness guard, and the
# fuzzy layer's partial_ratio independently scores a literal substring
# alignment as ~100% regardless of how short it is -- same vulnerability,
# different layer). Found via real degraded-audio testing: heavy
# pitch/tempo distortion sometimes makes a fine-tuned STT model collapse
# to a single bare, generic word instead of a full (if garbled) sentence
# -- e.g. Hindi "मुझे" ("me"), a literal substring of 11 of the 20 Hindi
# UDK phrases. That fragment carries no real distinguishing signal, but
# was firing an UNCONFIRMED TRIGGER_ALL (bypassing verify_by_default
# entirely) on whichever UDK happened to be first in iteration order --
# not a fluke: the same structural ambiguity exists in EVERY language's
# corpus, including English ("me" is a substring of 10/20 English UDKs,
# "help" of 5) -- it just hadn't manifested there yet because English's
# STT hasn't been observed collapsing this far under distortion. Fixed by
# checking whether the fragment is a substring of MORE THAN ONE UDK's
# phrase (ambiguous -- real evidence of genericness, not guesswork about
# which specific words are "filler") and capping confidence at this same
# KWS_MATCH_CONFIDENCE value if so: still registers as SOMETHING (this
# project's stated bias toward false positives over silent misses), but
# needs corroboration/repetition to escalate to TRIGGER_ALL, same as any
# other moderate-confidence signal -- instead of an unconditional full
# alarm on a coin-flip UDK. A fragment that's a substring of exactly ONE
# UDK is real, unambiguous evidence and keeps full confidence.
GENERIC_FRAGMENT_CONFIDENCE = KWS_MATCH_CONFIDENCE

# Fuzzy layer, third check (2026-09-25, 4.6 h of real Indian YouTube audio): 226 of 300
# speech false alarms were fuzzy matches of a long sentence that merely STARTS like a UDK --
# "I'm not saying this only because..." -> "I'm not safe" (53x), "Now, I want to start" ->
# "I want to go home now" (42x). partial_ratio and token_set_ratio both score these high,
# because the few shared words cover most of a short phrase. This check asks for the UDK's
# own words, in order, inside a stretch of the transcript about as long as the phrase; each
# word may be misheard (character similarity >= FUZZY_WORD_SIMILARITY: "plice" ~ "police").
FUZZY_MIN_WORD_COVERAGE = 0.75  # 3-word phrases need all 3 words, 4-7 words may miss one
FUZZY_WORD_SIMILARITY = 75


def _word_coverage(phrase: str, transcript: str) -> float:
    """Best fraction of the phrase's words found (fuzzy per word, any order -- "please help
    me" is "Help me, please") within any window of the transcript one word longer than the
    phrase. A transcript no longer than the phrase scores 1.0: it is judged whole, as
    before, since that is what a garbled shout looks like ("You do me!" was "Get away!",
    prepped_data pocket clip, caught only by the loose fuzzy match)."""
    p, t = phrase.split(), transcript.split()
    n = len(p)
    if len(t) <= n:
        return 1.0
    best = 0
    for start in range(len(t) - n):
        free = t[start:start + n + 1]
        hits = 0
        for a in p:
            j = next((j for j, b in enumerate(free) if fuzz.ratio(a, b) >= FUZZY_WORD_SIMILARITY), None)
            if j is not None:
                hits += 1
                free.pop(j)
        best = max(best, hits)
    return best / n


# A transcript made ONLY of these words can't trigger any layer (2026-09-25, user decision):
# a lone "Please." / "Right?" / "Now" / "I'm" / "The" carries no distress meaning, and such
# fragments caused one-word false alarms on real YouTube audio. Function words, fillers and
# politeness only -- command words ("get", "go", "let", "away", "back", "stop", "don't") and
# content words ("help", "police", "hurt", "scared") stay distinctive, so "Get away!",
# "Let go!" and "Help!" still alert. Forms are _normalize()d (apostrophes stripped).
NON_DISTINCTIVE_WORDS = frozenset(
    "please now right the a an i im me my you your he she it we they this that is am are was "
    "to of in on at here there and or so oh okay ok yeah yes well just what".split())


# Fragment rule (2026-09-26): a transcript no longer than a UDK phrase ("I don't", "I want",
# "You want to be here.") only matches that UDK if it contains one of the phrase's ANCHOR
# words -- its words minus function words, auxiliaries and modals. Such fragments matched
# as exact (a substring of one UDK) or fuzzy (judged whole) and carry no distress meaning;
# they were most of the real-world text false alarms ("I don't know." -> UDK_14 12x) and
# the negation clips' alarms. Anchors are compared per word with the fuzzy word similarity,
# so ASR noise ("plice") still counts. "Help!", "Get away!", "Stay back!" keep their anchor.
# UDK_FRAGMENT_FUZZY=0 applies it to exact fragments only (eval switch: the fuzzy part
# also drops garbled shouts with no anchor word, e.g. "You do me!" for "Let go of me").
FUNCTION_WORDS = NON_DISTINCTIVE_WORDS | frozenset(
    "do dont does doesnt did didnt want wants wanted need needs have has be been can cant will wont "
    "would should not no ill its for from being".split())  # for/from/being: "From my father." hit UDK_03
FRAGMENT_FUZZY = os.environ.get("UDK_FRAGMENT_FUZZY", "1") != "0"


def _has_anchor(phrase: str, transcript: str) -> bool:
    words = transcript.split()
    anchors = [w for w in phrase.split() if w not in FUNCTION_WORDS]
    # "don't" as a COMMAND ("Please don't!", "No, don't!") is itself a UDK word (UDK_06/13);
    # as a statement ("I don't know") it is a function word
    lead = next((w for w in words if w not in ("please", "no", "oh", "just")), "")
    if lead == "dont" and "dont" in phrase.split():
        return True
    return any(fuzz.ratio(a, w) >= FUZZY_WORD_SIMILARITY for a in anchors for w in words)


# Sentence boundaries in Whisper's punctuated output, for decide_all().
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# Real bug found and fixed: kws.py's spot() already collapses its DTW
# distance into a binary match/no-match at DEFAULT_MATCH_THRESHOLD=0.18,
# discarding HOW confidently it matched. A borderline distance right at
# that edge (0.167-0.172, measured on real audio) combined with a
# borderline STT confidence (0.755) to spuriously corroborate an ordinary
# sentence into an unconfirmed TRIGGER_ALL -- made worse by real
# wav2vec2/CPU inference nondeterminism occasionally flipping that exact
# distance across 0.18 on IDENTICAL audio between runs (the same
# phenomenon documented in README.md's "muffled" 19-vs-18 split).
# Corroboration specifically (not KWS-alone detection, which stays at the
# calibrated 0.18 -- that's the STT-outage safety net and should stay
# permissive) needs a stricter margin, calibrated against the same real
# corpus calibrate_kws_threshold.py already measured: at 0.15, recall
# drops from 93.9% to 81.8% but false-positive rate is CUT IN HALF (10.0%
# -> 5.0%). That asymmetry is exactly right for corroboration's specific
# role: missing a corroboration boost only means the detection stays at
# the still-safe, still-alerting TRIGGER_VERIFY tier (STT's own match
# stands on its own); a WRONG boost skips confirmation entirely. Being
# conservative here costs little and buys a lot.
KWS_CORROBORATION_MAX_DISTANCE = 0.15

# A THIRD independent signal, on top of STT text and KWS audio-embedding
# matching: scream_detector.py's pure-signal-processing acoustic
# distress score (roughness/pitch/energy -- no trained model). Real
# calibration there found only 57.8% recall / 9.0% FPR on its own
# (RAVDESS fear/angry speech vs. real negatives) -- nowhere near good
# enough to be a standalone trigger, which is exactly why it is wired
# the same way KWS corroboration already is: it can only ADD confidence
# to a segment where STT/KWS already independently produced a
# TRIGGER_VERIFY-or-better match (see decide()'s gate below, which runs
# BEFORE this is even consulted) -- never fire, or even get evaluated,
# on its own. A below-average detector that only ever corroborates an
# existing signal is still a net improvement; the same detector allowed
# to trigger alone would mean missing real distress speech more often
# than it catches it.


def _normalize(text: str) -> str:
    """Real bug found and fixed: the previous `[^\\w\\s]` regex relied on
    Python's `\\w`, which does NOT include Unicode combining marks
    (category Mn/Mc) -- the vowel signs (matras) that carry essential
    meaning in Devanagari/Telugu/Kannada/Tamil and other Brahmic scripts.
    That regex silently turned "मुझे" (Hindi "mujhe", = "me") into "मझ"
    (garbled, a different/meaningless string) by stripping its two vowel
    signs, which would have corrupted every non-Latin-script UDK phrase
    and transcript alike. Fixed by stripping actual punctuation/symbol
    Unicode categories (P*/S*) instead of relying on \\w's incomplete
    letter/mark distinction -- verified to strip English punctuation
    identically to the old regex while preserving Indic combining marks."""
    text = text.lower().strip()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class MatchResult:
    udk: UDK
    layer: Literal["exact", "fuzzy", "semantic", "kws", "none"]
    confidence: float  # 0.0-1.0


@dataclass
class DecisionEvent:
    decision: Decision
    udk: UDK | None
    confidence: float
    transcript: str
    layer: str
    repeated: bool
    reason: str


def match_transcript(transcript: str, udks: list[UDK], semantic_matcher: SemanticBackend | None = None) -> MatchResult:
    """Best match across all UDKs for one transcript, highest layer wins."""
    norm_transcript = _normalize(transcript)
    if not norm_transcript or set(norm_transcript.split()) <= NON_DISTINCTIVE_WORDS:
        return MatchResult(udk=None, layer="none", confidence=0.0)  # type: ignore[arg-type]

    norm_phrases = [_normalize(udk.phrase) for udk in udks]
    # See GENERIC_FRAGMENT_CONFIDENCE's comment: a transcript that's a
    # substring of more than one UDK's phrase is ambiguous, real evidence
    # of genericness rather than a specific match.
    fragment_is_ambiguous = sum(1 for p in norm_phrases if norm_transcript and norm_transcript in p) > 1

    best: MatchResult | None = None

    for udk, norm_phrase in zip(udks, norm_phrases):
        # Layer 1: exact (substring either direction, catches short
        # utterances like "help" matching inside a fuller phrase). Only
        # the "transcript is the short side, found inside a longer
        # phrase" direction is at risk of the generic-fragment problem --
        # true equality or the transcript containing the WHOLE phrase are
        # both unambiguous regardless of what else is in the corpus.
        if norm_phrase == norm_transcript or norm_phrase in norm_transcript:
            candidate = MatchResult(udk=udk, layer="exact", confidence=1.0)
            if best is None or candidate.confidence > best.confidence:
                best = candidate
            continue
        if norm_transcript in norm_phrase:
            if not _has_anchor(norm_phrase, norm_transcript):  # see FUNCTION_WORDS: "I don't" inside UDK_14
                continue
            confidence = GENERIC_FRAGMENT_CONFIDENCE if fragment_is_ambiguous else 1.0
            candidate = MatchResult(udk=udk, layer="exact", confidence=confidence)
            if best is None or candidate.confidence > best.confidence:
                best = candidate
            continue

        # Layer 2: fuzzy (ASR noise, word-order, minor transcription diffs).
        # Discounted 0.9x, same treatment as the semantic stub below --
        # found via evaluate_pipeline_corpus.py's real negative corpus:
        # partial_ratio's substring-alignment scoring inflates on short
        # UDK phrases against ordinary sentences that merely share a few
        # common words ("Call me later" hit 0.67 against no real UDK
        # relationship), 9/20 real negatives false-positived through this
        # layer specifically (0 through exact/semantic/KWS). The discount
        # still preserves the legitimate ASR-noise case ("call the plice
        # right now" -> 0.97 raw, 0.873 discounted, still clears
        # TRIGGER_ALL_THRESHOLD) while pushing most of the measured false
        # positives (clustered at 0.60-0.67) back below TRIGGER_VERIFY_THRESHOLD.
        # The raw gate is per-matcher (semantic.py's `fuzzy_threshold`,
        # default 0.55 -- same as English's calibrated value) because real
        # per-language testing found 0.55 far too permissive for Hindi/
        # Telugu/Kannada/Tamil specifically: false positives (0.65-0.81 raw)
        # and genuine matches (0.70-1.0 raw) overlap badly at 0.55 but
        # cleanly separate by 0.80 -- see semantic.py's module docstring.
        fuzzy_threshold = getattr(semantic_matcher, "fuzzy_threshold", 0.55) if semantic_matcher is not None else 0.55
        fuzzy_score = fuzz.partial_ratio(norm_phrase, norm_transcript) / 100.0
        # Real false positive found and fixed this session (real negative
        # podcast audio, not synthetic): partial_ratio's substring-alignment
        # scoring can land a short, common-word-heavy UDK phrase like "Let
        # go of me" at 0.667 raw against long, repetitive filler text
        # ("...going to go to the next video...") purely from coincidental
        # short-word alignment ("go"/"of"/"me" scattered nearby), not any
        # real relationship -- 0.667*0.9=0.600 lands exactly AT
        # TRIGGER_VERIFY_THRESHOLD instead of below it, the discount's
        # documented "pushes MOST, not all" gap. token_set_ratio (real
        # shared-WORD overlap, not character-substring alignment) cleanly
        # separates these: 0.29-0.34 on the false positives found vs. 0.627
        # on the documented legitimate ASR-noise case ("call the plice
        # right now" -> UDK_01, test_pipeline.py) -- same "two independent
        # signals must agree" principle this project already applies to
        # KWS+STT fusion and scream_score corroboration, applied here to
        # partial_ratio's own single-scorer blind spot.
        fuzzy_word_overlap = fuzz.token_set_ratio(norm_phrase, norm_transcript) / 100.0
        if (fuzzy_score >= fuzzy_threshold and fuzzy_word_overlap >= fuzzy_threshold
                and _word_coverage(norm_phrase, norm_transcript) >= FUZZY_MIN_WORD_COVERAGE
                and not (FRAGMENT_FUZZY and len(norm_transcript.split()) <= len(norm_phrase.split())
                         and not _has_anchor(norm_phrase, norm_transcript))):
            # partial_ratio scores a literal substring alignment as ~1.0
            # regardless of length -- same generic-fragment risk as the
            # exact layer above (see GENERIC_FRAGMENT_CONFIDENCE), so the
            # same ambiguity cap applies here too, not just there.
            raw_confidence = fuzzy_score * 0.9
            confidence = min(raw_confidence, GENERIC_FRAGMENT_CONFIDENCE) if fragment_is_ambiguous else raw_confidence
            candidate = MatchResult(udk=udk, layer="fuzzy", confidence=confidence)
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        # Layer 3: semantic -- a real embedding model if one was given,
        # else the lighter token-overlap stand-in (see module docstring).
        # The gating threshold comes from the matcher itself (default
        # 0.55 for anything that doesn't declare one, e.g. FakeSemantic
        # test doubles) -- semantic.py's real backends use different
        # models with different real-calibrated cutoffs per language
        # (0.55 for English, 0.80 for LaBSE/the other four languages;
        # see semantic.py's module docstring for why they can't share one).
        if semantic_matcher is not None:
            semantic_score = semantic_matcher.similarity(norm_phrase, norm_transcript)
            semantic_threshold = getattr(semantic_matcher, "threshold", 0.55)
            if semantic_score >= semantic_threshold:
                candidate = MatchResult(udk=udk, layer="semantic", confidence=semantic_score)
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        else:
            semantic_score = fuzz.token_set_ratio(norm_phrase, norm_transcript) / 100.0
            if semantic_score >= 0.55:
                # Discounted vs. real fuzzy/exact confidence, since it's a
                # weaker signal than a real embedding model gives.
                candidate = MatchResult(udk=udk, layer="semantic", confidence=semantic_score * 0.9)
                if best is None or candidate.confidence > best.confidence:
                    best = candidate

    return best or MatchResult(udk=None, layer="none", confidence=0.0)  # type: ignore[arg-type]


def _kws_match_to_result(kws_match: KWSMatch | None, udks: list[UDK]) -> MatchResult:
    if kws_match is None:
        return MatchResult(udk=None, layer="none", confidence=0.0)  # type: ignore[arg-type]
    udk = next((u for u in udks if u.udk_id == kws_match.phrase_id), None)
    if udk is None:  # KWS matched a phrase_id this engine doesn't know about -- ignore, don't guess
        return MatchResult(udk=None, layer="none", confidence=0.0)  # type: ignore[arg-type]
    return MatchResult(udk=udk, layer="kws", confidence=KWS_MATCH_CONFIDENCE)


def _fuse(stt: MatchResult, kws: MatchResult, kws_match: KWSMatch | None) -> tuple[MatchResult, bool]:
    """Best of the two independent branches, plus whether they agree.
    Agreement (both branches independently landing on the same UDK) is
    cross-validation, not just "two votes" -- treated like repetition
    already is: strong enough on its own to justify TRIGGER_ALL. Requires
    the raw KWS distance (not just the fixed-confidence MatchResult) to
    be within KWS_CORROBORATION_MAX_DISTANCE -- see that constant's
    comment for why a merely-borderline KWS match isn't trusted enough to
    justify skipping confirmation on its own."""
    agree = (
        stt.udk is not None
        and kws.udk is not None
        and stt.udk.udk_id == kws.udk.udk_id
        and kws_match is not None
        and kws_match.distance <= KWS_CORROBORATION_MAX_DISTANCE
    )
    best = stt if stt.confidence >= kws.confidence else kws
    return best, agree


class UDKEngine:
    """Stateful across a journey so it can apply the repetition booster
    (Section 6/7): tracks recent detections per udk_id within the last
    REPEAT_WINDOW_S seconds of journey time."""

    def __init__(self, udks: list[UDK], semantic_matcher: SemanticBackend | None = None):
        self.udks = udks
        self.semantic_matcher = semantic_matcher
        self._recent: dict[str, list[float]] = {}
        self._recent_distinct: list[tuple[float, str]] = []

    def decide_all(self, transcript: str, now_s: float | None = None, **kwargs) -> list[DecisionEvent]:
        """decide() on the whole segment, plus one extra event per SENTENCE
        that matches a different UDK on its own (2026-09-23, prepped_data:
        "Let go! ... Please let go! You're hurting me!" is one VAD segment,
        and decide() only ever reports its single best UDK, so UDK_13 was
        lost behind UDK_10). The first element is always the whole-segment
        decision, identical to decide(). Extra events go through decide()
        too, so two DIFFERENT UDKs in one segment hit the existing
        distinct-UDK escalation, same as two phrases said in a row."""
        now_s = now_s if now_s is not None else time.monotonic()
        primary = self.decide(transcript, now_s=now_s, **kwargs)
        events = [primary]
        sentences = [s for s in _SENTENCE_SPLIT.split(transcript.strip()) if s]
        if len(sentences) < 2:
            return events
        seen = {primary.udk.udk_id} if primary.decision != "NO_ACTION" and primary.udk else set()
        sentence_kwargs = {k: v for k, v in kwargs.items() if k != "kws_match"}  # KWS is a whole-segment signal
        # A sentence shorter than the shortest UDK (3 words for the current 20)
        # only counts if its words appear WORD FOR WORD, as whole words, in some
        # UDK. Real shouts pass ("Help!", "Please don't!" -> UDK_13, "Please!");
        # letter- or meaning-level guesses on fragments don't ("No." only hit
        # because "no" is inside "not"; "go to."; "Goodbye." via semantic).
        # 2026-09-23: a plain 3-word minimum cut FPR but dropped the "Help!" /
        # "Please don't!" triggers in a real distress clip, failing the rule
        # that no change may lose a successful trigger.
        norm_udks = [f" {_normalize(u.phrase)} " for u in self.udks]
        min_words = min(len(p.split()) for p in norm_udks)
        for sentence in sentences:
            words = _normalize(sentence).split()
            if len(words) < min_words and not any(f" {' '.join(words)} " in p for p in norm_udks):
                continue
            m = match_transcript(sentence, self.udks, semantic_matcher=self.semantic_matcher)
            if m.udk is None or m.udk.udk_id in seen or m.confidence < TRIGGER_VERIFY_THRESHOLD:
                continue
            event = self.decide(sentence, now_s=now_s, **sentence_kwargs)
            if event.decision != "NO_ACTION":
                seen.add(event.udk.udk_id)
                events.append(event)
        return events

    def snapshot(self) -> tuple:
        """Escalation-window state, so a caller can re-decide ONE segment
        (separation.recheck_verify_with_separation) without the first guess
        also counting as a second, distinct detection."""
        return {k: list(v) for k, v in self._recent.items()}, list(self._recent_distinct)

    def restore(self, state: tuple) -> None:
        recent, distinct = state
        self._recent, self._recent_distinct = {k: list(v) for k, v in recent.items()}, list(distinct)

    def _record_and_check_repeat(self, udk_id: str, now_s: float) -> bool:
        history = self._recent.setdefault(udk_id, [])
        history.append(now_s)
        # prune anything outside the window
        cutoff = now_s - REPEAT_WINDOW_S
        while history and history[0] < cutoff:
            history.pop(0)
        return len(history) >= 2  # this one plus at least one earlier

    def _record_and_check_distinct(self, udk_id: str, now_s: float) -> bool:
        """Tracks every UDK match across the whole engine (not per-udk_id
        like repetition does), so N different phrases firing within the
        window escalate even though no single one repeated."""
        self._recent_distinct.append((now_s, udk_id))
        cutoff = now_s - REPEAT_WINDOW_S
        self._recent_distinct = [(t, u) for t, u in self._recent_distinct if t >= cutoff]
        return len({u for _, u in self._recent_distinct}) >= DISTINCT_UDK_ESCALATION_COUNT

    def decide(
        self,
        transcript: str,
        now_s: float | None = None,
        kws_match: KWSMatch | None = None,
        scream_score: float | None = None,
        stt_suspect: bool = False,
        dual_asr_disagreement: bool = False,
    ) -> DecisionEvent:
        now_s = now_s if now_s is not None else time.monotonic()
        stt_result = match_transcript(transcript, self.udks, semantic_matcher=self.semantic_matcher)
        kws_result = _kws_match_to_result(kws_match, self.udks)
        match, corroborated = _fuse(stt_result, kws_result, kws_match)

        # STT-quality veto (stt_confidence_gate.py / dual_asr_guard.py):
        # discounts the text-match confidence BEFORE the hard floor below,
        # rather than adding a new independent gate -- same "no single
        # signal fires alone" shape as everything else here, just applied
        # in the other direction (subtracting confidence, not adding it).
        # Skipped when KWS independently agrees (corroborated) or when KWS
        # alone is winning (match.layer == "kws"): KWS never looks at the
        # STT transcript at all (it's audio-embedding+DTW), so a signal
        # about STT's text quality specifically says nothing about
        # whether KWS's own independent detection should be distrusted.
        effective_confidence = match.confidence
        veto_reason = None
        if (stt_suspect or dual_asr_disagreement) and not corroborated and match.layer != "kws":
            effective_confidence = min(match.confidence, TRIGGER_VERIFY_THRESHOLD - 0.01)
            veto_reason = (
                "text match vetoed by dual-ASR disagreement (independent second model didn't corroborate this transcript)"
                if dual_asr_disagreement
                else "text match vetoed by STT-confidence guardrail (decoder didn't trust its own output)"
            )

        # Gate runs BEFORE scream_score is even looked at -- a high
        # acoustic distress score alone, with no text/KWS match at all,
        # never reaches TRIGGER_VERIFY, let alone TRIGGER_ALL. See
        # SCREAM_SCORE_THRESHOLD's comment for why.
        if match.udk is None or effective_confidence < TRIGGER_VERIFY_THRESHOLD:
            return DecisionEvent(
                decision="NO_ACTION",
                udk=match.udk,
                confidence=effective_confidence,
                transcript=transcript,
                layer=match.layer,
                repeated=False,
                reason=veto_reason or "below TRIGGER_VERIFY threshold",
            )

        repeated = self._record_and_check_repeat(match.udk.udk_id, now_s)
        distinct_escalation = self._record_and_check_distinct(match.udk.udk_id, now_s)
        scream_corroborated = scream_score is not None and scream_score >= SCREAM_SCORE_THRESHOLD

        if (
            match.udk.verify_by_default
            and not repeated
            and not corroborated
            and not distinct_escalation
            and not scream_corroborated
        ):
            return DecisionEvent(
                decision="TRIGGER_VERIFY",
                udk=match.udk,
                confidence=match.confidence,
                transcript=transcript,
                layer=match.layer,
                repeated=repeated,
                reason="phrase is common-in-ordinary-conversation, verify by default (Section 16)",
            )

        if match.confidence >= TRIGGER_ALL_THRESHOLD or repeated or corroborated or distinct_escalation or scream_corroborated:
            reason = (
                "independent STT+KWS agreement"
                if corroborated
                else "multiple distinct UDKs triggered within window" if distinct_escalation
                else "repeated within window" if repeated
                else "acoustic distress signal corroborated the match" if scream_corroborated
                else "confidence >= TRIGGER_ALL threshold"
            )
            return DecisionEvent(
                decision="TRIGGER_ALL",
                udk=match.udk,
                confidence=match.confidence,
                transcript=transcript,
                layer=match.layer,
                repeated=repeated,
                reason=reason,
            )

        return DecisionEvent(
            decision="TRIGGER_VERIFY",
            udk=match.udk,
            confidence=match.confidence,
            transcript=transcript,
            layer=match.layer,
            repeated=repeated,
            reason="confidence between TRIGGER_VERIFY and TRIGGER_ALL thresholds",
        )
