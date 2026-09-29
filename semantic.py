"""
Real semantic/paraphrase matching (Section 4/17: "a real deployment uses
a pretrained sentence-embedding model... for paraphrase cases like
'he's going to hurt me' matching 'someone is threatening me'. That
model's weights are on huggingface.co" -- reachable now, so built here).

Compared against the rapidfuzz token_set_ratio stub udk_engine.py used
before, on 5 real English pairs (see README.md for the full table): the
real model is clearly better at rejecting unrelated phrases and
recognizing near-duplicate UDKs (much better separation on both),
roughly comparable on genuine paraphrases, and -- worth stating plainly
-- actually scores Section 4's own flagship example ("he's going to hurt
me" / "someone is threatening me") slightly *lower* than the stub did,
and neither clears the existing 0.55 threshold on that specific pair.
Three sentence-embedding models were compared on the same 5 pairs before
picking this one: paraphrase-multilingual-MiniLM-L12-v2 (used here for
English), all-MiniLM-L6-v2, and all-mpnet-base-v2. For English, the
0.55 threshold is kept as-is: real true-positive pairs (0.59-0.76) sit
comfortably above it and real true-negative pairs (0.15-0.26) sit
comfortably below.

CORRECTION, found while adding Hindi/Telugu/Kannada/Tamil support: this
model's "multilingual" claim from Section 17 was NEVER actually verified
against Telugu, Kannada, or Tamil specifically -- only assumed from its
name. Real testing (12 translated pairs) found it works well for Hindi
(clean separation, comparable to English) but is NEAR-USELESS for the
three Dravidian languages at 0.55: completely UNRELATED sentences
("I am hungry" vs. "someone is trying to hurt me") scored 0.56-0.89,
all clearing the threshold. That's not a calibration problem, it's the
model's embeddings not discriminating meaning at all in those languages
-- almost certainly because they're far lower-resource than Hindi/
English in its training data.

`sentence-transformers/LaBSE` (Language-agnostic BERT Sentence
Embeddings, explicitly trained for broad cross-lingual retrieval across
109 languages) was tested as a replacement and performs far better: a
real 10-positive/200-negative-pair calibration per language (paraphrases
of 10 UDKs x the same 20 ordinary-negative sentences already used
throughout this project, translated) found clean, consistent separation
across all four languages at threshold 0.80 (100% recall, 0.5% FPR,
uniformly -- see README.md for the full per-language table). LaBSE's
raw similarity scores run systematically higher than the MiniLM model's
even for genuine matches (0.86-0.99 vs. 0.59-0.76), which is exactly why
this needed its own re-calibrated threshold rather than reusing 0.55 --
NOT swept the way kws.py's threshold was swept in as much depth (10
positives, not calibrate_kws_threshold.py's ~120 clips), so treat 0.80 as
a real, evidence-based starting point, not a finished calibration.
English keeps the original model + 0.55 (already validated, not touched,
no reason to risk regressing it); LaBSE + 0.80 is used only for
`SUPPORTED_LANGUAGES` other than English (see `udks.py`).

CORRECTION #2, found while investigating why Telugu/Kannada/Tamil's real
degraded-condition FPR (25-35%) ran well above Hindi's (15%, same as
English) despite sharing this same LaBSE/0.80 setup: the culprit wasn't
LaBSE at all -- it was `udk_engine.py`'s FUZZY layer (rapidfuzz
`partial_ratio`), whose raw 0.55 gate was tuned once against English
negatives and never re-checked for the other four languages. Real
per-language measurement (translate 20 ordinary negatives via NLLB,
synthesize via MMS-TTS, transcribe with each language's real fine-tuned
STT, take the best `partial_ratio` against all 20 UDK phrases) found
false positives clustering at 0.65-0.81 raw ratio for ALL FOUR languages
including Hindi (not just the three Dravidian ones), while genuine UDK
audio scores 0.70-1.0 -- the two distributions overlap badly at 0.55 but
cleanly separate by 0.80: FPR at 0.55 measured 60%/60%/40%/65% (hi/te/
kn/ta); at 0.80 it drops to 0%/0%/0%/5%, with fuzzy-layer recall
essentially unaffected (any single item that stops clearing the fuzzy
gate is still well within LaBSE's own 0.80 semantic gate, since it's the
same phrase with minor ASR noise, not a genuine paraphrase). Fixed by
giving the fuzzy layer its own per-matcher `fuzzy_threshold`, same
pattern as `threshold` -- English keeps 0.55 (untouched, already
validated), the shared multilingual matcher uses `MULTILINGUAL_FUZZY_THRESHOLD`.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# 0.55 -> 0.65 (2026-09-23, trace_semantic_fp.py): below 0.60 a semantic
# match can't reach TRIGGER_VERIFY anyway, and 0.60 -> 0.65 loses 0/32 real
# paraphrases while cutting real-negative-segment false alarms 1/156 -> 0,
# ordinary 4/20 -> 3/20, adversarial 5/20 -> 4/20. 0.70 would cost 5/32
# paraphrases. See trace_semantic_fp_output.log.
DEFAULT_THRESHOLD = 0.65
DEFAULT_FUZZY_THRESHOLD = 0.55

# See module docstring's "CORRECTION" section for the real calibration
# behind this. Language codes match udks.py's SUPPORTED_LANGUAGES.
MULTILINGUAL_MODEL = "sentence-transformers/LaBSE"
MULTILINGUAL_THRESHOLD = 0.80
# See module docstring's "CORRECTION #2" -- real per-language sweep found
# false positives (0.65-0.81 raw partial_ratio) and genuine matches
# (0.70-1.0) overlap at 0.55 but cleanly separate by 0.80, for all four
# non-English languages (Hindi included, not just the Dravidian three).
MULTILINGUAL_FUZZY_THRESHOLD = 0.80


class SemanticBackend(Protocol):
    def similarity(self, a: str, b: str) -> float: ...
    threshold: float
    fuzzy_threshold: float


class SentenceTransformerSemanticMatcher:
    """Real backend. Deferred import, same reasoning as stt.py's
    FasterWhisperSTT and kws.py's Wav2Vec2DTWSpotter -- sentence-
    transformers pulls in torch, a much heavier dependency than anything
    a caller who doesn't need real semantic matching should have to pay
    for. `threshold` and `fuzzy_threshold` are per-instance (not shared
    module constants) specifically so English (DEFAULT_*) and the other
    four languages (MULTILINGUAL_*) can each use their own
    real-calibrated cutoffs -- see module docstring."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.model_version = f"sentence-transformers-{model_name}"
        self.threshold = threshold
        self.fuzzy_threshold = fuzzy_threshold
        # match_transcript compares one transcript against every UDK phrase,
        # and decide_all() does it per sentence too -- without this, the same
        # 20 phrases and the same transcript were re-encoded on every call.
        # Bounded so a long-running journey process can't grow it forever.
        import functools

        self._encode = functools.lru_cache(maxsize=512)(
            lambda text: self._model.encode(text, convert_to_tensor=True)
        )

    def similarity(self, a: str, b: str) -> float:
        from sentence_transformers import util

        return util.cos_sim(self._encode(a), self._encode(b)).item()
