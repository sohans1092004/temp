"""
Real calibration for scream_detector.py's SCREAM_SCORE_THRESHOLD --
same discipline as calibrate_kws_threshold.py: measure recall/FPR on
real audio, don't ship an unmeasured starting-point threshold (that
mistake already cost this project once, with KWS's original 0.30
giving a 97.5% false-positive rate).

Positives (stated honestly -- neither is a literal blood-curdling
scream, both are the closest real, obtainable proxies):
  - RAVDESS real human actors, fear/angry emotion, STRONG intensity --
    raised-voice, high-arousal delivery, not silent-film screaming.
  - The real Vakeel Saab abduction clip's own genuine on-screen
    screaming ("Aaaah! No, no, no, no, no!" and "Ah! Ah! Ah!" segments,
    found during this project's own real-world audio testing -- see
    README.md's "Real-world audio testing" section), extracted
    directly by timestamp. This is the one genuinely real scream in
    this project's real_recordings collection.

Negatives: RAVDESS neutral-emotion clips (calm real speech) plus this
project's own real_recordings/negatives/ clips (real podcasts/shorts,
various languages, deliberately NOT re-using the positives folder).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

from scream_detector import SAMPLE_RATE, ScreamDetector
from vad import VAD

REPO_ID = "birgermoell/ravdess"
ACTORS = [f"Actor_{i:02d}" for i in range(1, 9)]
EMOTION_CODES = {"neutral": "01", "fear": "06", "angry": "05"}
STATEMENTS = ["01", "02"]
REPETITIONS = ["01", "02"]
REAL_RECORDINGS_DIR = Path(__file__).parent / "real_recordings"

# The real scream clip's own segment boundaries, from this project's
# earlier VAD run against _converted_positives_2.wav (see README.md) --
# (1.95, 12.93) covers "Balla Levate. Wait no. Aaaah! No, no, no, no, no!"
REAL_SCREAM_CLIP = REAL_RECORDINGS_DIR / "positives" / "_converted_positives_2.wav"
REAL_SCREAM_WINDOW_S = (1.95, 12.93)


def _load_pcm_16k_mono(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        n_channels, framerate = w.getnchannels(), w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if framerate != SAMPLE_RATE:
        duration = len(samples) / framerate
        src_x = np.linspace(0, duration, num=len(samples), endpoint=False)
        dst_x = np.linspace(0, duration, num=int(duration * SAMPLE_RATE), endpoint=False)
        samples = np.interp(dst_x, src_x, samples)
    return samples.astype(np.int16).tobytes()


def _filename(emotion_code: str, statement: str, repetition: str, actor_num: str, intensity: str) -> str:
    return f"03-01-{emotion_code}-{intensity}-{statement}-{repetition}-{actor_num}.wav"


def collect_ravdess(emotion: str, intensity: str) -> list[bytes]:
    code = EMOTION_CODES[emotion]
    clips = []
    for actor in ACTORS:
        actor_num = actor.split("_")[1]
        for statement in STATEMENTS:
            for repetition in REPETITIONS:
                fname = _filename(code, statement, repetition, actor_num, intensity)
                try:
                    local_path = hf_hub_download(REPO_ID, f"{actor}/{fname}", repo_type="dataset")
                except Exception:
                    continue
                clips.append(_load_pcm_16k_mono(local_path))
    return clips


def collect_real_negatives() -> list[bytes]:
    vad = VAD()
    clips = []
    for path in sorted((REAL_RECORDINGS_DIR / "negatives").glob("_converted_negatives_*.wav")):
        audio, sr = sf.read(str(path), dtype="int16")
        assert sr == SAMPLE_RATE
        pcm = audio.tobytes()
        for seg in vad.segment_speech(pcm):
            clips.append(seg.pcm)
    return clips


def collect_real_scream() -> bytes:
    audio, sr = sf.read(str(REAL_SCREAM_CLIP), dtype="int16")
    assert sr == SAMPLE_RATE
    start_sample = int(REAL_SCREAM_WINDOW_S[0] * sr)
    end_sample = int(REAL_SCREAM_WINDOW_S[1] * sr)
    return audio[start_sample:end_sample].tobytes()


def main() -> None:
    detector = ScreamDetector()

    print("Downloading RAVDESS neutral (real calm speech, negatives)...")
    neutral_clips = collect_ravdess("neutral", intensity="01")
    print("Downloading RAVDESS fear/angry, STRONG intensity (real raised-voice proxy, positives)...")
    fear_clips = collect_ravdess("fear", intensity="02")
    angry_clips = collect_ravdess("angry", intensity="02")

    print("Loading this project's own real negative recordings (podcasts/shorts)...")
    real_negative_segments = collect_real_negatives()

    print("Extracting the real scream segment from the Vakeel Saab clip...")
    real_scream_pcm = collect_real_scream()

    neg_scores = [detector.score(pcm) for pcm in neutral_clips] + [detector.score(pcm) for pcm in real_negative_segments]
    fear_scores = [detector.score(pcm) for pcm in fear_clips]
    angry_scores = [detector.score(pcm) for pcm in angry_clips]
    real_scream_score = detector.score(real_scream_pcm)

    print(f"\n{'=' * 70}")
    print(f"Negatives (n={len(neg_scores)}): RAVDESS-neutral={len(neutral_clips)} + real-negatives={len(real_negative_segments)}")
    print(f"  mean={np.mean(neg_scores):.3f}  median={np.median(neg_scores):.3f}  p90={np.percentile(neg_scores, 90):.3f}  max={np.max(neg_scores):.3f}")
    print(f"Fear (n={len(fear_scores)}, RAVDESS strong):")
    print(f"  mean={np.mean(fear_scores):.3f}  median={np.median(fear_scores):.3f}  min={np.min(fear_scores):.3f}")
    print(f"Angry (n={len(angry_scores)}, RAVDESS strong):")
    print(f"  mean={np.mean(angry_scores):.3f}  median={np.median(angry_scores):.3f}  min={np.min(angry_scores):.3f}")
    print(f"\nReal Vakeel Saab scream segment score: {real_scream_score:.3f}")

    print(f"\n{'=' * 70}")
    print("THRESHOLD SWEEP (recall on fear+angry-strong vs FPR on all negatives)")
    print("=" * 70)
    all_pos = fear_scores + angry_scores
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        recall = sum(1 for s in all_pos if s >= threshold) / len(all_pos)
        fpr = sum(1 for s in neg_scores if s >= threshold) / len(neg_scores)
        real_scream_caught = "YES" if real_scream_score >= threshold else "no"
        print(f"  threshold={threshold:.2f}  recall={recall:.1%}  FPR={fpr:.1%}  real_scream_caught={real_scream_caught}")


if __name__ == "__main__":
    main()
