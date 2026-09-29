"""SER (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, arousal/
dominance/valence regression) as a candidate distress signal -- arousal
is the dimension most directly relevant to distress (high arousal =
heightened physiological activation, present in both fear and anger).
Evaluated on the exact clips used for BEATs' own calibration, plus an
explicit false-positive check on the full 163-segment real negatives
corpus (same corpus used for BEATs' own threshold work this session).

REAL LOADING BUG found and fixed before trusting any output: the
model's own README-documented `EmotionModel.from_pretrained(...)` call
hits a genuine transformers-version incompatibility in this environment
(`AttributeError: 'EmotionModel' object has no attribute
'all_tied_weights_keys'` -- an internal API renamed in newer
transformers than this model card's code predates) inside
`from_pretrained`'s own finalization step, unrelated to anything in this
project. `AutoModelForAudioClassification.from_pretrained(...)` "works"
(no exception) but silently loads the WRONG classifier head --
`classifier.weight`/`projector.weight` come back MISSING (randomly
initialized) while the checkpoint's real head
(`classifier.dense`/`classifier.out_proj`, a 2-layer regression head)
is reported UNEXPECTED and discarded. Either path would have produced
confident-looking garbage without erroring. Fixed by constructing the
model from config and loading `model.safetensors`'s state dict directly
via `load_state_dict(strict=False)`, bypassing the buggy
`from_pretrained` finalization entirely -- verified zero missing/zero
unexpected keys before trusting any score below.

VERDICT: DEAD END -- real signal on RAVDESS alone, but fails the
explicit false-positive check the task required. Real numbers:
  real scream (genuine):     arousal=0.663
  RAVDESS fear (n=32):        mean=0.849  (min=0.709, max=0.986)
  RAVDESS angry (n=32):       mean=0.917  (min=0.730, max=1.013)
  RAVDESS calm (n=32):        mean=0.394
  RAVDESS happy (n=32):       mean=0.558
  real negatives (n=163):     mean=0.544  max=0.871
    fraction >= 0.7: 17.2%  (at/above the WEAKEST genuine fear/angry
                              RAVDESS example)
Real ordinary podcast/conversational speech averages HIGHER arousal
(0.544) than RAVDESS calm (0.394) and is comparable to RAVDESS happy
(0.558) -- "arousal" here appears to track ordinary speaking energy/
enthusiasm (animated hosts, engaged conversation) rather than genuine
distress specifically. No threshold choice fixes this: any cutoff low
enough to catch the weakest real fear/angry example (~0.71-0.73) also
catches ~17% of ordinary real negative speech. Not wired in.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from beats_distress_detector import BEATsDistressDetector
from calibrate_scream_detector import collect_ravdess, collect_real_negatives
from vad import VAD

SAMPLE_RATE = 16_000
REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        return self.classifier(hidden_states)  # (arousal, dominance, valence)


class SERArousalScorer:
    model_version = "audeering-wav2vec2-large-robust-msp-dim"

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(REPO)
        self._model = EmotionModel(config)
        weights_path = hf_hub_download(REPO, "model.safetensors")
        state_dict = load_file(weights_path)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        assert not missing and not unexpected, f"weight loading mismatch: missing={missing} unexpected={unexpected}"
        self._model.eval()
        self._processor = Wav2Vec2Processor.from_pretrained(REPO)

    def dims(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> tuple[float, float, float]:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        if len(audio) < SAMPLE_RATE // 10:
            return 0.0, 0.0, 0.0
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values)
        arousal, dominance, valence = logits[0].tolist()
        return arousal, dominance, valence

    def score(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        """Arousal alone, as the distress-relevant dimension."""
        arousal, _, _ = self.dims(pcm, sample_rate)
        return arousal


def stats(name: str, values: list[float]) -> None:
    arr = np.array(values)
    print(f"  {name:28s} n={len(arr):3d} min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}")


def main() -> None:
    print("Loading BEATs (comparison) + SER...")
    beats = BEATsDistressDetector()
    ser = SERArousalScorer()

    print("Extracting real scream (correctly-windowed 2s peak)...")
    import librosa

    audio, sr = librosa.load(
        "real_recordings/positives/Nivetha Thomas Gets Abducted Vakeel Saab Malayalam Pawan Kalyan #YTShorts.mp3",
        sr=16000, mono=True,
    )
    start, end = int(45.0 * sr), int(47.0 * sr)
    scream_pcm = np.clip(audio[start:end] * 32767, -32768, 32767).astype(np.int16).tobytes()
    print(f"  SER dims on real scream: arousal/dominance/valence = {ser.dims(scream_pcm)}")

    print("Downloading RAVDESS fear/angry (strong) + calm/happy (normal)...")
    fear = collect_ravdess("fear", intensity="02")
    angry = collect_ravdess("angry", intensity="02")
    import calibrate_scream_detector as csd
    csd.EMOTION_CODES["calm"] = "02"
    csd.EMOTION_CODES["happy"] = "03"
    calm = csd.collect_ravdess("calm", intensity="01")
    happy = csd.collect_ravdess("happy", intensity="01")

    groups = {
        "RAVDESS fear (strong)": fear,
        "RAVDESS angry (strong)": angry,
        "RAVDESS calm": calm,
        "RAVDESS happy": happy,
    }
    print(f"\n{'=' * 90}\nSER arousal vs BEATs score() -- same RAVDESS clips\n{'=' * 90}")
    for label, clips in groups.items():
        beats_scores = [beats.score(pcm) for pcm in clips]
        ser_scores = [ser.score(pcm) for pcm in clips]
        print(f"\n--- {label} ---")
        stats("BEATs", beats_scores)
        stats("SER arousal", ser_scores)

    print(f"\n{'=' * 90}\nFALSE-POSITIVE CHECK: SER arousal on the full real negatives corpus (n=163 segments,\nsame corpus used for BEATs' own threshold work this session)\n{'=' * 90}")
    import os

    vad = VAD()

    def load_pcm(path, sr=16000):
        a, _ = librosa.load(path, sr=sr, mono=True)
        return np.clip(a * 32767, -32768, 32767).astype(np.int16).tobytes()

    neg_dir = "real_recordings/negatives"
    arousal_scores = []
    for fname in sorted(os.listdir(neg_dir)):
        path = os.path.join(neg_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            pcm = load_pcm(path)
        except Exception:
            continue
        for seg in vad.segment_speech(pcm):
            arousal_scores.append(ser.score(seg.pcm))
    arr = np.array(arousal_scores)
    print(f"  n={len(arr)} real negative segments")
    print(f"  arousal: min={arr.min():.4f} mean={arr.mean():.4f} median={np.median(arr):.4f} max={arr.max():.4f}")
    for t in [0.5, 0.6, 0.7, 0.8]:
        print(f"  fraction >= {t}: {100*np.mean(arr >= t):.1f}%")


if __name__ == "__main__":
    main()
