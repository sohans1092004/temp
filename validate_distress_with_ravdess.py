"""
Second, independent real-human-speech check on the same question
validate_distress_with_tess.py already answered: does real distress
delivery actually cost FasterWhisperSTT as much accuracy as the synthetic
`pitch_tempo_distorted` condition (evaluate_pipeline_corpus.py) suggests
(40-45% recall vs. 100% baseline)?

TESS has a real limitation this dataset doesn't share: only 2 speakers,
both older Canadian actresses, saying single words. RAVDESS (Ryerson
Audio-Visual Database of Emotional Speech and Song) is a different real
human dataset -- 24 professional actors (12 male, 12 female), two fixed
carrier SENTENCES ("Kids are talking by the door." / "Dogs are sitting by
the door."), each recorded in 8 real emotions at 2 intensities. Using a
different dataset, different speakers (gender-balanced, not just 2
actresses), and full sentences instead of single words is a genuinely
independent check, not a re-run of the same evidence.

Same controlled-comparison design as the TESS script: for each
(actor, statement, repetition), the exact same words are compared across
neutral vs. fear vs. angry delivery -- only the emotion differs. Ground
truth is known from the fixed carrier sentences, not from any model.

This tests STT robustness to real emotional delivery specifically (does
the transcript contain each ground-truth word), not the full
VAD->STT->KWS->fusion UDK pipeline -- like the TESS script, it can't
replace evaluate_pipeline_corpus.py, only inform whether that corpus's
synthetic distress proxy is pessimistic, optimistic, or roughly right.

Real result (see README.md for the table): consistent with the TESS
finding, not contradicting it.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from stt import FasterWhisperSTT
from vad import VAD, SAMPLE_RATE

REPO_ID = "birgermoell/ravdess"
ACTORS = [f"Actor_{i:02d}" for i in range(1, 9)]  # 4 male, 4 female (odd/even actor numbers)
EMOTION_CODES = {"neutral": "01", "fear": "06", "angry": "05"}
STATEMENTS = {
    "01": ["kids", "are", "talking", "by", "the", "door"],
    "02": ["dogs", "are", "sitting", "by", "the", "door"],
}
INTENSITY = "01"  # normal -- neutral has no "strong" variant, so this keeps all three comparable
REPETITIONS = ["01", "02"]


def _filename(emotion_code: str, statement: str, repetition: str, actor_num: str) -> str:
    return f"03-01-{emotion_code}-{INTENSITY}-{statement}-{repetition}-{actor_num}.wav"


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


def main() -> None:
    print(f"Downloading matched RAVDESS clips for {len(ACTORS)} actors (real human speech)...")
    vad = VAD()
    stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    results: dict[str, list[float]] = {e: [] for e in EMOTION_CODES}
    misses: dict[str, list[str]] = {e: [] for e in EMOTION_CODES}

    for actor in ACTORS:
        actor_num = actor.split("_")[1]
        for statement, gt_words in STATEMENTS.items():
            for repetition in REPETITIONS:
                for emotion, code in EMOTION_CODES.items():
                    fname = _filename(code, statement, repetition, actor_num)
                    try:
                        local_path = hf_hub_download(REPO_ID, f"{actor}/{fname}", repo_type="dataset")
                    except Exception as e:
                        print(f"  skip {actor}/{fname}: {e}")
                        continue
                    pcm = _load_pcm_16k_mono(local_path)
                    segments = vad.segment_speech(pcm)
                    if not segments:
                        results[emotion].append(0.0)
                        misses[emotion].append(f"{actor}/{fname}: no speech detected")
                        continue
                    seg = max(segments, key=lambda s: s.end_ms - s.start_ms)
                    transcript = stt.transcribe(seg.pcm).text.lower()
                    hits = sum(1 for w in gt_words if w in transcript)
                    word_accuracy = hits / len(gt_words)
                    results[emotion].append(word_accuracy)
                    if word_accuracy < 1.0:
                        misses[emotion].append(f"{actor}/{fname}: transcript={transcript!r} ({hits}/{len(gt_words)} words)")

    print(f"\n{'=' * 70}")
    print(f"STT WORD-RECOGNITION ACCURACY BY EMOTION (real human speech, RAVDESS, n={len(results['neutral'])} clips each)")
    print("=" * 70)
    for emotion in EMOTION_CODES:
        acc = sum(results[emotion]) / len(results[emotion])
        print(f"{emotion:<10}{acc:>8.1%}")

    print("\nMisses (clips with at least one missed word), by emotion:")
    for emotion in EMOTION_CODES:
        if misses[emotion]:
            print(f"  {emotion}: {len(misses[emotion])}/{len(results[emotion])}")
            for m in misses[emotion]:
                print(f"    {m}")


if __name__ == "__main__":
    main()
