"""
Keyword spotting without a training dataset (Section 4's Approach D/E,
deferred in Section 17's v1 decision for exactly this reason: "no public
dataset combines these specific phrases + distress delivery... has to be
built, not found").

The answer here isn't building that dataset -- it's query-by-example
spoken term detection: embed a handful of reference clips per phrase with
an existing pretrained speech encoder (no training, no dataset of our
own), then match incoming audio against them via DTW over the frame-level
embeddings.

Two other pretrained approaches were tried first and rejected on real
evidence, not assumption -- see README.md for the numbers:
  - CLAP zero-shot text-audio matching: the unrelated clip's top text
    match was the SAME phrase the UDK clip matched, with only a ~2%
    margin between them. Not usable.
  - CLAP audio-to-audio, mean-pooled: WORSE -- a different UDK phrase
    scored higher than the same phrase reworded. CLAP's embedding space
    is tuned for broad scene/caption matching ("a dog barking"), not
    fine-grained spoken-word content.
  - Wav2Vec2 (speech-specific, phonetically-aware) mean-pooled: got the
    ordering right (same phrase > different phrase > unrelated) but the
    margins were thin (~2%), because mean-pooling over the whole
    utterance collapses timing information and lets voice/prosody
    dominate.
  - Wav2Vec2 + DTW over frame-level features (this module): preserves
    temporal alignment instead of collapsing to one vector -- same
    phrase reworded scored a DTW distance of 0.22, a different UDK
    phrase 0.27, unrelated speech 0.36. Real, usable separation.

This is a real, independent detection path (Section 4's point: doesn't
share STT's failure modes, doesn't need STT to succeed at all).

DEFAULT_MATCH_THRESHOLD is calibrated against a measured corpus
(calibrate_kws_threshold.py), not picked from a handful of examples --
see that file and README.md for the full methodology and numbers. Short
version: 80 positive clips (20 general UDKs x exact phrase + a natural
paraphrase x 2 TTS voices NOT used for the reference bank) and 40
negative clips (20 ordinary-conversation phrases x 2 voices). The old
starting-point threshold of 0.30, picked from 4 example clips, turned out
to be badly miscalibrated once measured properly: it gave a 97.5% false
positive rate on ordinary speech. Still not Section 12's full corpus --
no real distress delivery, no muffling, no real human speakers, English
only, one TTS engine -- so this is a real, measured, much-better-grounded
number, not a claim that Section 12's testing layer is complete.

One limitation the calibration surfaced that threshold tuning can't fix:
6-7 of the positive clips were closest to a DIFFERENT real UDK phrase
entirely, not to their own (e.g. "I don't feel safe at all" landed
closer to UDK_14 "I don't feel safe here" than to its own UDK_02 "I'm not
safe"; similarly UDK_07/UDK_04 and UDK_08/UDK_17). These are phrases that
are genuinely similar to each other, not to ordinary speech -- so the
practical effect is "the right kind of alert with the wrong specific
phrase_id/transcript logged," not a missed detection or a false one
mixed in with ordinary conversation. Worth knowing about, not something
this module tries to disambiguate further.

Checked for the "generic fragment" false-confidence bug found and fixed
in udk_engine.py's exact/fuzzy text-matching layers (a short, generic
transcript like "मुझे"/"me" scored full 1.0 confidence via naive
substring/partial_ratio containment against a much longer phrase --
GENERIC_FRAGMENT_CONFIDENCE) -- this module does NOT share it. Real test
(a short, generic word synthesized and spotted against the real English
and Hindi reference banks): "मुझे" (the exact fragment that caused the
text-layer bug) scored a DTW distance of 0.342 against its closest
reference, "मैं" scored 0.541 -- both roughly 2-2.7x the 0.20 match
threshold, cleanly rejected, not a false match. Why this doesn't
transfer: DTW's distance is length-normalized over the FULL alignment
path (`dp[n,m]/max(n,m)` in `_dtw_distance`), so a short query aligned
against a much longer reference accumulates real cost representing the
reference's unmatched content -- there's no equivalent to a text
substring check's "found it somewhere, call it a perfect match regardless
of length" free pass. The vulnerability was specific to the text layers'
scoring mechanics (containment / local-alignment-percentage), not
something DTW-over-frame-embeddings inherits.

Multiple references per phrase (enroll_kws_references.py enrolls all 3
installed voices per general UDK, not just 1) -- tried specifically to
improve recall on the distress-sim/overlapping-speech conditions
evaluate_pipeline_corpus.py measured as weak (40-50%). Measured result,
stated honestly: it didn't work. Whole-pipeline recall on those two
conditions barely moved (45%/50% vs. 40%/50%, within noise for n=20),
while the KWS layer's OWN false-positive rate on calm ordinary speech
roughly doubled (7.5% to 15% at the same threshold) from having more
reference embeddings for an unrelated utterance to coincidentally land
close to. The reason the hoped-for improvement didn't materialize: STT
*and* KWS both operate on the same degraded audio segment -- more
reference voices doesn't help when the incoming audio itself is too
distorted (pitch/tempo-shifted, or mixed with a second speaker) for
wav2vec2's embedding to represent well in the first place. The
bottleneck for those two conditions is the audio signal, not reference-
voice coverage. Kept anyway for the separate, real value of voice/accent
robustness on clean-ish speech (Section 12 asks for that too) -- just
not oversold as having fixed the degraded-condition gap it was tried
for. The end-to-end false-positive rate didn't measurably worsen despite
the isolated KWS-layer regression (the fusion logic absorbed it in this
corpus), but that's a second-order finding, not the headline one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16_000

# Calibrated against calibrate_kws_threshold.py's measured corpus (see
# the module docstring for the full history). Current numbers, measured
# against the 3-voice reference bank with paraphrase-only positives
# (exact-phrase positives can't be used for calibration anymore -- with
# all 3 installed voices now enrolled, they'd just self-match):
#   threshold  recall  false_pos_rate
#     0.15      81.8%      5.0%
#     0.18      93.9%     15.0%   <- picked (was 94.6%/7.5% on the 1-voice bank)
#     0.20      97.0%     25.0%
#     0.22     100.0%     35.0%
#     0.30     100.0%     97.5%   <- the original, unmeasured "starting point"
# Kept at 0.18 despite the higher false-positive rate: recall is
# essentially unchanged from the 1-voice bank, and a false match here
# still only costs a confirmation prompt (KWS_MATCH_CONFIDENCE lands in
# TRIGGER_VERIFY range in udk_engine.py, not TRIGGER_ALL, unless STT
# agrees or it repeats) -- but see the module docstring for why the
# multi-voice bank didn't actually fix what it was tried for, which is
# the more important finding than this threshold number.
DEFAULT_MATCH_THRESHOLD = 0.18

# Multilingual KWS (English's model/approach doesn't transfer as-is --
# see below). Real finding: a raw self-supervised multilingual model
# (facebook/wav2vec2-large-xlsr-53) gives ZERO usable signal for this
# DTW approach -- 0/10 real paraphrases matched their own phrase as
# closest, positive/negative distances completely overlapping
# (0.000-0.025 for both). Root cause: this approach needs a model
# fine-tuned for ASR (sharpens word/phrase-level structure), not just
# self-supervised pretrained -- English's own model
# (facebook/wav2vec2-base-960h) already IS ASR-fine-tuned, which is WHY
# it works, not incidental. The Vakyansh project's per-language
# ASR-fine-tuned wav2vec2 models give a real, usable signal instead.
# Real per-language calibration (10 paraphrase positives + 20 negatives
# each -- much lighter than English's 120-clip corpus, a real starting
# point not a finished calibration):
#   Language  Correct-closest  Threshold  Recall  FPR
#   Telugu    8/10             0.25       80%     15%
#   Kannada   7/10             0.20       60%     30%
#   Hindi     8/10             0.20       90%     10%   <- close to English's own quality
#   Tamil     7/10             0.25       70%     25%
INDIC_KWS_MODELS: dict[str, str] = {
    "te": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
    "kn": "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560",
    "hi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
    "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
}
INDIC_KWS_THRESHOLDS: dict[str, float] = {
    "te": 0.25,
    "kn": 0.20,
    "hi": 0.20,
    "ta": 0.25,
}


@dataclass
class KWSMatch:
    phrase_id: str
    distance: float  # lower = more similar; see DEFAULT_MATCH_THRESHOLD


class KWSBackend(Protocol):
    def enroll(self, phrase_id: str, pcm: bytes) -> None: ...
    def spot(self, pcm: bytes) -> KWSMatch | None: ...
    def clear(self, phrase_id: str) -> None: ...


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: [T, D] L2-normalized frame embeddings. Cost per frame pair
    is 1 - cosine similarity; standard DTW DP; normalized by path length
    so clip duration doesn't dominate the score."""
    n, m = len(a), len(b)
    cost = 1.0 - a @ b.T
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    # Anti-diagonal (wavefront) order: every cell with i + j == k depends only
    # on diagonals k-1 and k-2, so a whole diagonal is one numpy op -- ~n+m
    # vector steps instead of n*m Python steps (was ~5 s of a ~7 s clip, times
    # 60 references). Identical result to the cell-by-cell loop (test_kws.py).
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        dp[i, j] = cost[i - 1, j - 1] + np.minimum(np.minimum(dp[i - 1, j], dp[i, j - 1]), dp[i - 1, j - 1])
    return dp[n, m] / max(n, m)



class Wav2Vec2DTWSpotter:
    """Real backend: a pretrained wav2vec2 encoder (no training data of
    our own required) + DTW query-by-example matching. Deferred import,
    same reasoning as stt.py's FasterWhisperSTT -- transformers/torch are
    a much heavier dependency than anything else in this project, no
    reason to pay that cost for callers who only need MockKWS."""

    def __init__(self, model_name: str = DEFAULT_MODEL, match_threshold: float = DEFAULT_MATCH_THRESHOLD,
                 device: str = "cpu"):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self._device = device  # "cuda" moves the wav2vec2 encoder to GPU; DTW stays numpy
        self._model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self._model.eval()
        # Wav2Vec2FeatureExtractor (audio preprocessing only), not the
        # full Wav2Vec2Processor (feature extractor + CTC tokenizer) --
        # this class only ever uses the model's hidden states for DTW,
        # never decodes text, so the tokenizer half is dead weight. Real
        # bug found adding multilingual support: raw self-supervised
        # checkpoints like facebook/wav2vec2-large-xlsr-53 (never
        # fine-tuned for CTC/ASR) have no vocab.json at all, so
        # Wav2Vec2Processor.from_pretrained() crashes outright on them --
        # facebook/wav2vec2-base-960h (the English default) only worked
        # before because it happens to ALSO ship a CTC tokenizer from its
        # own ASR fine-tuning, not because one was needed.
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.match_threshold = match_threshold
        self.model_version = f"wav2vec2-dtw-{model_name.rsplit('/', 1)[-1]}"
        # Multiple references per phrase_id (voice/delivery variants) --
        # spot() takes the best (minimum-distance) reference per phrase,
        # so enrolling more variants of a phrase can only help recall
        # (more chances for a real match), never hurt it. The tradeoff,
        # measured, not assumed: more references also means more chances
        # for an unrelated utterance to land close to *one* of them by
        # chance -- see enroll_kws_references.py/README.md for the
        # before/after recall and false-positive-rate numbers.
        self._references: dict[str, list[np.ndarray]] = {}

    def _frame_embeddings(self, pcm: bytes) -> np.ndarray:
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self._device)
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
        return hidden / np.linalg.norm(hidden, axis=1, keepdims=True)

    def enroll(self, phrase_id: str, pcm: bytes) -> None:
        """pcm is a reference clip for this phrase -- TTS-generated is
        fine (Section 17's own recommended bootstrap for narrow-vocabulary
        KWS with no natural corpus); a real recording works the same way.
        Adds to this phrase's reference set rather than replacing it --
        call clear(phrase_id) first if you actually want to replace
        (Section 5: re-enrollment should deactivate the old phrase, not
        just add to it)."""
        self._references.setdefault(phrase_id, []).append(self._frame_embeddings(pcm))

    def clear(self, phrase_id: str) -> None:
        self._references.pop(phrase_id, None)

    def spot(self, pcm: bytes) -> KWSMatch | None:
        if not self._references:
            return None
        query = self._frame_embeddings(pcm)
        best_id, best_distance = min(
            ((pid, min(_dtw_distance(query, ref) for ref in refs)) for pid, refs in self._references.items()),
            key=lambda x: x[1],
        )
        if best_distance > self.match_threshold:
            return None
        return KWSMatch(phrase_id=best_id, distance=best_distance)

    def save_references(self, path) -> None:
        """Persists enrolled reference embeddings so a real deployment
        doesn't need TTS/Windows/any specific OS at runtime -- generate
        once (enroll_kws_references.py), load anywhere. Each reference is
        flattened to its own npz key (phrase_id__index) since a phrase
        can now have several, of different lengths."""
        flat = {f"{pid}__{i}": ref for pid, refs in self._references.items() for i, ref in enumerate(refs)}
        np.savez_compressed(path, **flat)

    def load_references(self, path) -> None:
        with np.load(path) as data:
            references: dict[str, list[np.ndarray]] = {}
            for key in data.files:
                phrase_id, _, _index = key.rpartition("__")
                references.setdefault(phrase_id, []).append(data[key])
            self._references = references


class MockKWS:
    """Test double, same pattern as stt.py's MockSTT: returns whatever
    match was registered for the exact pcm bytes, instead of doing real
    embedding/DTW work."""

    def __init__(self):
        self._pcm_to_match: dict[bytes, KWSMatch] = {}
        self.model_version = "mock-kws-0"

    def register(self, pcm: bytes, phrase_id: str, distance: float = 0.1) -> None:
        self._pcm_to_match[pcm] = KWSMatch(phrase_id=phrase_id, distance=distance)

    def enroll(self, phrase_id: str, pcm: bytes) -> None:
        # A clip matched against its own enrolled reference is exactly
        # what the real backend gives (distance ~0) -- treating the
        # enrollment clip as a registered self-match keeps the mock
        # useful for testing "was this reference actually enrolled"
        # instead of being a pure no-op.
        self.register(pcm, phrase_id, distance=0.0)

    def spot(self, pcm: bytes) -> KWSMatch | None:
        return self._pcm_to_match.get(pcm)

    def clear(self, phrase_id: str) -> None:
        self._pcm_to_match = {pcm: m for pcm, m in self._pcm_to_match.items() if m.phrase_id != phrase_id}
