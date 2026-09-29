"""
Same measurement as measure_real_distress_shift.py (real pitch/tempo
shift between neutral and fear/angry delivery, same speaker, same words),
run against RAVDESS instead of TESS -- a second independent dataset,
24 gender-balanced professional actors instead of 2 actresses, full
sentences instead of single words. See validate_distress_with_ravdess.py
for why this comparison matters and the STT-accuracy side of it.
"""

from __future__ import annotations

import wave

import librosa
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "birgermoell/ravdess"
ACTORS = [f"Actor_{i:02d}" for i in range(1, 9)]
EMOTION_CODES = {"neutral": "01", "fear": "06", "angry": "05"}
STATEMENTS = ["01", "02"]
REPETITIONS = ["01", "02"]
INTENSITY = "01"


def _load_float(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _mean_f0(audio: np.ndarray, sr: int) -> float | None:
    f0, _voiced_flag, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.mean(f0)) if len(f0) else None


def main() -> None:
    semitone_shifts: dict[str, list[float]] = {"fear": [], "angry": []}
    duration_ratios: dict[str, list[float]] = {"fear": [], "angry": []}

    for actor in ACTORS:
        actor_num = actor.split("_")[1]
        for statement in STATEMENTS:
            for repetition in REPETITIONS:
                def path_for(emotion_code: str) -> str:
                    fname = f"03-01-{emotion_code}-{INTENSITY}-{statement}-{repetition}-{actor_num}.wav"
                    return hf_hub_download(REPO_ID, f"{actor}/{fname}", repo_type="dataset")

                try:
                    neutral_audio, sr = _load_float(path_for(EMOTION_CODES["neutral"]))
                except Exception:
                    continue
                neutral_f0 = _mean_f0(neutral_audio, sr)
                neutral_dur = len(neutral_audio) / sr
                if neutral_f0 is None:
                    continue
                for emotion in ["fear", "angry"]:
                    try:
                        audio, _ = _load_float(path_for(EMOTION_CODES[emotion]))
                    except Exception:
                        continue
                    f0 = _mean_f0(audio, sr)
                    if f0 is None:
                        continue
                    semitone_shifts[emotion].append(12 * np.log2(f0 / neutral_f0))
                    duration_ratios[emotion].append(neutral_dur / (len(audio) / sr))

    for emotion in ["fear", "angry"]:
        st, dr = semitone_shifts[emotion], duration_ratios[emotion]
        print(f"{emotion}: mean pitch shift = {np.mean(st):+.2f} semitones (n={len(st)}), mean tempo ratio = {np.mean(dr):.2f}x (n={len(dr)})")


if __name__ == "__main__":
    main()
