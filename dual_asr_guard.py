"""Cross-model disagreement guardrail for the "confidently wrong"
failure class stt_confidence_gate.py structurally cannot catch (see that
module's docstring): a transcript where faster-whisper's own decoding
metadata (temperature=0.0, an unremarkable avg_logprob) gives no sign
anything is wrong, yet the text is not what was actually said, and
happens to clear udk_engine.py's TRIGGER_VERIFY_THRESHOLD.

No metric internal to ONE model can ever flag this -- by construction,
"confidently wrong" means the model's own self-assessment is exactly
what a genuine correct transcript would also show. The only way to
catch it is an independent second opinion. wav2vec2 (facebook/
wav2vec2-base-960h, CTC-based, no encoder-decoder, different training
data/objective from Whisper) is the model actually validated this
session -- NOT NeMo Parakeet, which was the first choice considered and
rejected: nemo_toolkit[asr] requires lightning<=2.4.0,>2.2.1, a direct
version conflict with the pytorch-lightning<1.5.0 that pyannote.audio/
speechbrain/asteroid already share and depend on in this environment
(see conv_tasnet_separator.py's own history of that exact dependency
fragility). wav2vec2 was already partially cached from this project's
own KWS work and needed no new dependency at all.

Real validated evidence this session -- 3 of 3 known real false-UDK-
trigger cases found this session were independently caught:
  clip_04 ch_a: whisper "I think it's time for us to go back home."
    (-> false match: UDK_18 "I want to go home now", conf=0.655) vs.
    wav2vec2 "" (empty)                              agreement=0.000
  clip_08 mono: whisper "We should order more category of me."
    (-> false match: UDK_10 "Let go of me", conf=0.704) vs.
    wav2vec2 "we shon't don't da won't coa me"        agreement=0.418
  clip_08 ch_b: whisper "We should all be more careful of me."
    (-> false match: UDK_10 "Let go of me", conf=0.663) vs.
    wav2vec2 "westo do daoaa"                         agreement=0.280
All three: agreement well below the 0.5 threshold below (fixed BEFORE
this check was run against these cases, not tuned to them -- see
AGREEMENT_THRESHOLD's comment). Two genuine failure branches were
distinguished, not assumed: does the second model recover the right
answer (0/5 tested -- it never did), or does it just disagree usefully
(5/5 -- garbled/empty output, structurally unlike Whisper's fluent wrong
guesses, no correlated-error pattern found). This is a SUPPRESSION
signal only, by design and by evidence -- it has ZERO measured recall
benefit, it only ever discounts a match, never independently produces
or upgrades one.

Cost: a second full model inference call, not free like
stt_confidence_gate.py. Must run selectively, never on every segment --
see api.py's wiring, which only invokes this when a segment has already
produced a UDK match at TRIGGER_VERIFY_THRESHOLD or above AND the
primary transcript's temperature is exactly 0.0 (the specific gap
stt_confidence_gate.py leaves open) -- the same "retry, don't replace"
cost discipline separation.py's NO_ACTION-only fallback already uses,
adapted to a narrower trigger condition targeting this specific gap
rather than duplicating that fallback's coverage.

REAL, UNRESOLVED FALSE-VETO RISK found in a full-corpus validation run
(not the n=5 spot-check above) -- a structural problem with wav2vec2-
base-960h as the second model, not a threshold to retune. On the 18
clean, scripted, ground-truth clips in real_recordings/mixed/ (exact
UDK phrases), this guard deterministically vetoed 2 GENUINE correct
detections, every run, on noisy/muffled conditions specifically:
  clip_14_noisy:   whisper "I'm in danger." (CORRECT, matches UDK_16)
                   vs. wav2vec2 "i in vengo"              -> false veto
  clip_15_muffled: whisper "Somebody help me now." (CORRECT, UDK_17)
                   vs. wav2vec2 "come modi hail mee now"  -> false veto
That's 2 false vetoes out of only 8 total invocations across the whole
60-clip corpus -- a 25% false-veto rate among this guard's own triggers,
concentrated exactly in the degraded-audio conditions this product cares
about most.

TWO candidate fixes were tried and BOTH FALSIFIED BY MEASUREMENT, not
just reasoned about -- recorded here so a future session doesn't retry
either without new evidence:
  1. Gate on wav2vec2's OWN self-confidence (mean max-softmax-probability
     across output frames) before trusting a disagreement. Measured:
     wav2vec2 reports 0.95-0.96 confidence on BOTH garbled false-veto
     outputs above -- a raw CTC argmax confidence reflects "how sure am I
     about each frame's most likely token," not "is my overall
     transcription correct," so it cannot distinguish a confidently-wrong
     model from a genuinely-uncertain one. No threshold on this signal
     works.
  2. A stricter/different AGREEMENT_THRESHOLD. Measured: the false-veto
     agreement scores (0.417, 0.419) sit INSIDE the range of the 3
     genuine validated catches (0.000, 0.418, 0.280) -- clip_14's 0.417
     is essentially identical to clip_08 mono's real catch at 0.418. No
     threshold on token_set_ratio separates these populations; they are
     not separable by score.
Root cause: wav2vec2-base-960h (no language model, older architecture)
simply garbles moderately-degraded real audio -- REGARDLESS of whether
Whisper's transcript of that same audio is right or wrong. "Disagreement"
therefore stops meaning "Whisper is probably wrong" and starts meaning
"this audio is somewhat degraded," which is true of a large fraction of
real-world noisy/muffled speech this system needs to handle correctly,
not a rare edge case.

STATUS: NOT recommended for enabling (UDK_ENABLE_DUAL_ASR=1) in anything
resembling production use until this is actually fixed -- e.g. a
genuinely stronger/more robust second model (one with its own language
model or comparable robustness to Whisper on degraded audio), or an
entirely different discrimination signal than model-vs-model text
agreement. Kept in the codebase, opt-in and off by default, as a real,
working prototype of the *idea* -- the "confidently wrong" gap it
targets is real and still open, this specific implementation is just not
safe to turn on yet.
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz

SAMPLE_RATE = 16_000

# Fixed in conversation BEFORE being run against any known case (see
# module docstring) -- an unremarkable, round threshold chosen to avoid
# unconsciously tuning it to the cases it was about to be judged against.
AGREEMENT_THRESHOLD = 0.5


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class Wav2Vec2DisagreementGuard:
    """Real backend. Deferred import, same reasoning as every other real
    backend in this project -- transformers/torch shouldn't cost a
    caller who never enables UDK_ENABLE_DUAL_ASR any startup time."""

    model_version = "wav2vec2-base-960h-disagreement-guard"

    def __init__(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        import torch

        audio = _pcm_to_float(pcm)
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0].strip().lower()

    def disagrees(self, primary_transcript: str, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """True if the second model's independent transcript doesn't
        agree with the primary one -- a suppression signal, see module
        docstring. Word-overlap ratio, not exact match: two real ASR
        models differ in punctuation/capitalization/minor phrasing even
        when both are right, so exact string comparison would false-flag
        genuine agreement constantly."""
        secondary = self.transcribe(pcm, sample_rate)
        agreement = fuzz.token_set_ratio(primary_transcript.lower(), secondary) / 100.0
        return agreement < AGREEMENT_THRESHOLD


def demo() -> None:
    """Smallest runnable sanity check: the model loads and runs without
    crashing, and the agreement math itself is correct on trivial cases.
    Distinguishing real disagreement on real degraded audio needs real
    audio (see module docstring's 3/3 validated cases) -- not a synthetic
    check, same reasoning as beats_distress_detector.py's demo()."""
    sr = SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    pcm = (np.clip(silence, -1, 1) * 32767).astype("<i2").tobytes()

    guard = Wav2Vec2DisagreementGuard()
    text = guard.transcribe(pcm)
    assert isinstance(text, str), "transcribe() must return a string even on silence"

    # Agreement math sanity, independent of the model/audio: identical
    # text must always agree, two clearly unrelated phrases must always
    # disagree. Real degraded-audio behavior is the module docstring's
    # 3/3 validated cases, not re-derived here with a toy example.
    identical_agreement = fuzz.token_set_ratio("get away from me", "get away from me") / 100.0
    unrelated_agreement = fuzz.token_set_ratio("get away from me", "quarterly revenue report") / 100.0
    assert identical_agreement >= AGREEMENT_THRESHOLD, "identical text must agree"
    assert unrelated_agreement < AGREEMENT_THRESHOLD, "unrelated text must disagree"
    print(f"[PASS] model loads and runs (silence -> {text!r}); agreement math sane")


if __name__ == "__main__":
    demo()
