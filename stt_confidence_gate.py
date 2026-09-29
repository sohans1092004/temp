"""STT self-confidence guardrail: flags a transcript as suspect using
metadata faster-whisper already computes per segment (avg_logprob,
temperature) but this project previously discarded entirely (see
stt.py's Transcript -- these fields were added specifically for this
module). No new model, no new inference call -- this is free.

Real evidence this session (see the conversation's temperature-fallback
investigation): re-running the SAME degraded/separated audio produced a
DIFFERENT fabricated sentence every time when Whisper's internal decoder
had rejected its own greedy-decode attempt and fallen back to stochastic
sampling (temperature > 0) -- the model's own decoder flagging "I didn't
trust my first attempt" before anyone downstream ever saw the output.
avg_logprob independently drops well below any genuine-correct baseline
on the worst fabrication case.

Calibrated on n=8 (this session's real sample, not a larger corpus):
  genuinely-hallucinating segments (temperature fallback fired):
    clip_08 (separated channel, multi-turn fabricated dialogue): avg_logprob
      -1.227 and -2.175 across two separate runs, temperature=1.0 every time
    clip_16 (separated channel, repeated-sentence hallucination): avg_logprob
      -0.894/-0.968, temperature=0.2 on the flagged segment
    clip_14 (mono, noisy-condition wrong transcript): avg_logprob=-1.250,
      temperature=1.0
  genuine-correct baselines (clip_01/09/13, mono clean clips): avg_logprob
    range -0.472 to -0.673, temperature=0.0 every time

THRESHOLDS ARE NOT FINAL -- calibrated on this n=8 sample specifically,
not a larger corpus, and should be re-validated before being trusted as
production-final, same caveat as every other real-but-small-n threshold
in this project (e.g. scream_detector.py's original SCREAM_SCORE_THRESHOLD).

KNOWN, STRUCTURAL LIMITATION, not a bug to fix here: this gate cannot
catch "confidently wrong" transcripts -- real, measured cases this
session (clip_04 mono/ch_a/ch_b, clip_08 mono/ch_b) all show
temperature=0.0 and an avg_logprob (-0.576 to -0.920) indistinguishable
from the genuine-correct baseline range above, despite being wrong
content that in 3/5 cases went on to clear udk_engine.py's own
TRIGGER_VERIFY_THRESHOLD via the text-matching layers. No metric
internal to one model's own decoding can ever flag output the model
itself is confidently, deterministically wrong about -- that failure
class is what dual_asr_guard.py's cross-model disagreement check is
for, not a gap in this module to close.
"""

from __future__ import annotations

from stt import Transcript

# See module docstring for the real n=8 evidence behind these two numbers.
AVG_LOGPROB_SUSPECT_THRESHOLD = -1.0
TEMPERATURE_SUSPECT_THRESHOLD = 0.0


def is_suspect(transcript: Transcript) -> bool:
    """True if this transcript's own decoding metadata suggests the model
    didn't trust its output (temperature fallback fired) or scored it
    unusually low (avg_logprob). None fields (a backend that doesn't
    populate them, e.g. MockSTT) are treated as "no signal" -- never
    flagged, never assumed confident either."""
    if transcript.avg_logprob is not None and transcript.avg_logprob < AVG_LOGPROB_SUSPECT_THRESHOLD:
        return True
    if transcript.temperature is not None and transcript.temperature > TEMPERATURE_SUSPECT_THRESHOLD:
        return True
    return False


def demo() -> None:
    """Runnable check using this session's real measured values (see
    module docstring) as ground truth, not synthetic numbers."""
    genuinely_hallucinating = [
        Transcript(text="x", model_version="t", avg_logprob=-1.227, temperature=1.0),
        Transcript(text="x", model_version="t", avg_logprob=-2.175, temperature=1.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.968, temperature=0.2),
        Transcript(text="x", model_version="t", avg_logprob=-1.250, temperature=1.0),
    ]
    genuine_correct = [
        Transcript(text="x", model_version="t", avg_logprob=-0.574, temperature=0.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.673, temperature=0.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.472, temperature=0.0),
    ]
    confidently_wrong = [
        Transcript(text="x", model_version="t", avg_logprob=-0.733, temperature=0.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.576, temperature=0.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.730, temperature=0.0),
        Transcript(text="x", model_version="t", avg_logprob=-0.920, temperature=0.0),
    ]
    unknown = Transcript(text="x", model_version="mock")

    assert all(is_suspect(t) for t in genuinely_hallucinating), "missed a real hallucination case"
    assert not any(is_suspect(t) for t in genuine_correct), "false-flagged a genuine correct transcript"
    assert not is_suspect(unknown), "a backend with no metadata must not be flagged"
    # Documented limitation, asserted so it can't silently change unnoticed:
    assert not any(is_suspect(t) for t in confidently_wrong), (
        "confidently-wrong cases should NOT be caught by this gate -- if this now fails, "
        "the module docstring's limitation claim is out of date, go update it"
    )
    print("[PASS] flags genuine hallucinations, leaves genuine-correct and no-metadata cases alone, "
          "confirms the documented confidently-wrong blind spot")


if __name__ == "__main__":
    demo()
