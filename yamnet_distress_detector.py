"""YAMNet (TensorFlow Hub, AudioSet, 521 classes) as a candidate
distress-cue classifier -- evaluated as a lighter/faster alternative to
beats_distress_detector.py, NOT a replacement decision yet. Separate,
additive investigation track -- does not touch or reopen the
overlapping-speech/UDK guardrail work frozen earlier this session.

Real, verified class indices (YAMNet's own yamnet_class_map.csv, not
guessed): Screaming=11, Shout=6, Yell=9, Children shouting=10,
Crying/sobbing=19, Baby cry=20, Groan=33, Speech=0, Conversation=2,
Laughter=13, Belly laugh=17, Whispering=12.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000

VOCAL_DISTRESS_INDICES = {
    11: "Screaming",
    6: "Shout",
    9: "Yell",
    10: "Children shouting",
    19: "Crying, sobbing",
    20: "Baby cry, infant cry",
    33: "Groan",
}
LAUGHTER_INDICES = {13: "Laughter", 17: "Belly laugh", 14: "Baby laughter"}
SPEECH_INDICES = {0: "Speech", 2: "Conversation"}


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class YAMNetDistressDetector:
    model_version = "yamnet-audioset-v1"

    def __init__(self):
        import tensorflow_hub as hub

        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

    def class_scores(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Mean-pooled per-class score across the clip's frames -- same
        clip-level aggregation BEATs' score() effectively does (mean over
        time before a single composite number), for direct comparability."""
        audio = _pcm_to_float(pcm)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        if len(audio) < SAMPLE_RATE // 10:
            return np.zeros(521, dtype=np.float32)
        scores, _embeddings, _spectrogram = self._model(audio)
        return scores.numpy().mean(axis=0)

    def score(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        """Max probability across the curated vocal-distress classes --
        same OR-not-sum reasoning as beats_distress_detector.py's score()."""
        probs = self.class_scores(pcm, sample_rate)
        return float(max(probs[idx] for idx in VOCAL_DISTRESS_INDICES))

    def per_class(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
        probs = self.class_scores(pcm, sample_rate)
        out = {}
        for group in (VOCAL_DISTRESS_INDICES, LAUGHTER_INDICES, SPEECH_INDICES):
            for idx, name in group.items():
                out[name] = float(probs[idx])
        return out


def demo() -> None:
    sr = SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    pcm = (np.clip(silence, -1, 1) * 32767).astype("<i2").tobytes()

    detector = YAMNetDistressDetector()
    score = detector.score(pcm)
    assert 0.0 <= score <= 1.0, f"score out of range: {score}"
    assert score < 0.3, f"1s of digital silence scored suspiciously high as distress: {score:.3f}"
    print(f"[PASS] YAMNet loads, runs, and scores silence low: {score:.3f}")


if __name__ == "__main__":
    demo()
