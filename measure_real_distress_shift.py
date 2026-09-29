"""
Measures the REAL pitch and duration shift between neutral and fear/angry
TESS clips of the same word, same speaker -- the evidence behind
audio_augment.py's and evaluate_pipeline_corpus.py's finding that naive
digital pitch-shift + time-stretch is NOT a valid distress-register
proxy, at any parameter setting (see those files' docstrings and
README.md for the full argument).

Real result: fear shifts pitch up by ~+8.05 semitones and speeds up
~1.28x relative to neutral (angry: ~+5.56 semitones, ~1.31x) -- a LARGER
pitch shift than this project's synthetic "distress-sim" condition used
(+3 semitones), yet validate_distress_with_tess.py shows real fear/angry
speech costs FasterWhisperSTT only ~3-4 accuracy points, while the
synthetic version (smaller shift) collapsed recall by ~55-60 points. The
gap isn't the magnitude, it's that digital pitch-shifting introduces
artifacts a real higher-pitched voice never has.
"""

from __future__ import annotations

import io
import wave
import zipfile
from pathlib import Path

import librosa
import numpy as np
from huggingface_hub import hf_hub_download

ACTOR = "OAF"  # older actress -- pick one for a consistent-speaker comparison
N_WORDS = 30


def _load_float(zf: zipfile.ZipFile, name: str) -> tuple[np.ndarray, int]:
    raw = zf.read(name)
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _mean_f0(audio: np.ndarray, sr: int) -> float | None:
    f0, _voiced_flag, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.mean(f0)) if len(f0) else None


def main() -> None:
    print("Downloading TESS (real human emotional speech, one-time)...")
    zip_path = hf_hub_download("myleslinder/tess", "data/tess.zip", repo_type="dataset")
    zf = zipfile.ZipFile(zip_path)

    by_word: dict[str, dict[str, str]] = {}
    for name in zf.namelist():
        parts = Path(name).stem.split("_")
        if len(parts) == 3 and parts[0] == ACTOR:
            by_word.setdefault(parts[1], {})[parts[2]] = name

    words = sorted(w for w, e in by_word.items() if all(x in e for x in ["neutral", "fear", "angry"]))[:N_WORDS]

    semitone_shifts: dict[str, list[float]] = {"fear": [], "angry": []}
    duration_ratios: dict[str, list[float]] = {"fear": [], "angry": []}

    for word in words:
        neutral_audio, sr = _load_float(zf, by_word[word]["neutral"])
        neutral_f0 = _mean_f0(neutral_audio, sr)
        neutral_dur = len(neutral_audio) / sr
        if neutral_f0 is None:
            continue
        for emotion in ["fear", "angry"]:
            audio, _ = _load_float(zf, by_word[word][emotion])
            f0 = _mean_f0(audio, sr)
            if f0 is None:
                continue
            semitone_shifts[emotion].append(12 * np.log2(f0 / neutral_f0))
            duration_ratios[emotion].append(neutral_dur / (len(audio) / sr))  # >1 = emotion clip is faster

    for emotion in ["fear", "angry"]:
        st, dr = semitone_shifts[emotion], duration_ratios[emotion]
        print(f"{emotion}: mean pitch shift = {np.mean(st):+.2f} semitones (n={len(st)}), mean tempo ratio = {np.mean(dr):.2f}x (n={len(dr)})")


if __name__ == "__main__":
    main()
