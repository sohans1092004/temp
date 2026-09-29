"""CLAP (laion/clap-htsat-unfused) as a candidate distress classifier,
scored via natural-language prompts rather than a fixed AudioSet class
list -- evaluated on the exact same clips used for BEATs' own
calibration this session, for a direct, controlled comparison.

CLAP's feature extractor requires 48kHz audio (BEATs/YAMNet use 16kHz) --
resampled per-call, not a design choice, just this model's requirement.

VERDICT: DEAD END, and not just "worse" -- inverted in the direction
that matters most. Real numbers (max distress-prompt probability vs.
BEATs' score()):
  real scream (genuine):    BEATs=0.454   CLAP=0.200
  RAVDESS fear (n=32):       BEATs mean=0.052   CLAP mean=0.482
  RAVDESS angry (n=32):      BEATs mean=0.084   CLAP mean=0.520
  RAVDESS calm (n=32):       BEATs mean=0.004   CLAP mean=0.169
  RAVDESS happy (n=32):      BEATs mean=0.007   CLAP mean=0.210
  real negatives (n=46):     BEATs mean=0.003 max=0.014   CLAP mean=0.277 max=0.917
CLAP scores a real ordinary negative podcast clip (0.917) far higher
than the one genuine real scream in this project's whole corpus (0.200)
-- it does not separate distress from calm at all on real audio, and
appears to react to something else entirely (loudness, prosody
variance, recording characteristics) uncorrelated with genuine
distress. Not wired in, not pursued further -- the natural-language-
prompt framing did not produce the hoped-for cleaner class semantics;
if anything the fixed-class BEATs classifier is far more reliable on
this real corpus than the flexible prompt-based one.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from beats_distress_detector import BEATsDistressDetector
from calibrate_scream_detector import collect_ravdess, collect_real_negatives

CLAP_SAMPLE_RATE = 48_000

DISTRESS_PROMPTS = [
    "a person screaming in fear",
    "someone crying in distress",
    "a person shouting for help",
    "a person in genuine physical danger",
]
NEUTRAL_PROMPTS = [
    "a person speaking calmly",
    "ordinary conversation between people",
    "someone talking normally",
    "a person laughing happily",
]


class ClapDistressScorer:
    model_version = "clap-htsat-unfused"

    def __init__(self):
        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self._model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self._model.eval()
        self._prompts = DISTRESS_PROMPTS + NEUTRAL_PROMPTS
        self._n_distress = len(DISTRESS_PROMPTS)

    def score(self, pcm: bytes, sample_rate: int = 16_000) -> float:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != CLAP_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=CLAP_SAMPLE_RATE)
        if len(audio) < CLAP_SAMPLE_RATE // 10:
            return 0.0
        inputs = self._processor(audio=audio, text=self._prompts, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._model(**inputs)
        probs = out.logits_per_audio.softmax(dim=-1)[0]
        # max distress-prompt probability vs. the whole prompt set (distress+neutral) --
        # a direct analog to BEATs' score() (max across a curated class set), same reasoning.
        return float(probs[: self._n_distress].max())


def stats(name: str, scores: list[float]) -> None:
    arr = np.array(scores)
    print(f"  {name:28s} n={len(arr):3d} min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}")


def main() -> None:
    print("Loading BEATs (for direct comparison) + CLAP...")
    beats = BEATsDistressDetector()
    clap = ClapDistressScorer()

    print("Extracting real scream (correctly-windowed 2s peak, established earlier this session)...")
    audio, sr = librosa.load(
        "real_recordings/positives/Nivetha Thomas Gets Abducted Vakeel Saab Malayalam Pawan Kalyan #YTShorts.mp3",
        sr=16000, mono=True,
    )
    start, end = int(45.0 * sr), int(47.0 * sr)
    scream_pcm = np.clip(audio[start:end] * 32767, -32768, 32767).astype(np.int16).tobytes()

    print("Downloading RAVDESS fear/angry (strong) + calm/happy (normal)...")
    fear = collect_ravdess("fear", intensity="02")
    angry = collect_ravdess("angry", intensity="02")
    import calibrate_scream_detector as csd
    csd.EMOTION_CODES["calm"] = "02"
    csd.EMOTION_CODES["happy"] = "03"
    calm = csd.collect_ravdess("calm", intensity="01")
    happy = csd.collect_ravdess("happy", intensity="01")

    print("Loading real negative segments...")
    negatives = collect_real_negatives()

    groups = {
        "real scream (correctly-windowed)": [scream_pcm],
        "RAVDESS fear (strong)": fear,
        "RAVDESS angry (strong)": angry,
        "RAVDESS calm": calm,
        "RAVDESS happy": happy,
        "real negatives": negatives,
    }

    print(f"\n{'=' * 90}\nCLAP vs BEATs -- same clips, same methodology\n{'=' * 90}")
    for label, clips in groups.items():
        beats_scores = [beats.score(pcm) for pcm in clips]
        clap_scores = [clap.score(pcm) for pcm in clips]
        print(f"\n--- {label} ---")
        stats("BEATs", beats_scores)
        stats("CLAP", clap_scores)

    print(f"\n{'=' * 90}\nReference (established this session): BEATs real scream=0.454, RAVDESS fear mean=0.052/max=0.188,")
    print("real negative ceiling=0.014")


if __name__ == "__main__":
    main()
