"""
Validates (or invalidates) the synthetic distress-sim recall number from
evaluate_pipeline_corpus.py (40-45%) against REAL human distress-adjacent
speech, not another synthetic proxy.

The TESS dataset (Toronto Emotional Speech Set -- 2 actresses, 200 fixed
target words, 7 emotions each, real human recordings) lets us compare
real FasterWhisperSTT transcription accuracy on the SAME word, said by
the SAME speaker, across neutral vs. fear vs. angry delivery -- a
controlled comparison no synthetic augmentation can give, because the
ground-truth word is known from the filename (`{actor}_{word}_{emotion}.wav`,
each spoken in the carrier phrase "Say the word ___").

This tests STT robustness to real emotional delivery specifically (word-
level transcription accuracy), not the full VAD->STT->KWS->fusion UDK
pipeline -- TESS's words aren't our UDK phrases, so it can't replace
evaluate_pipeline_corpus.py, only inform whether that corpus's synthetic
distress proxy was pessimistic, optimistic, or roughly right.
"""

from __future__ import annotations

import wave
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from stt import FasterWhisperSTT
from vad import VAD, SAMPLE_RATE

N_WORDS = 30  # words to sample, matched across all three emotions
ACTOR = "OAF"  # older actress -- pick one for a consistent-speaker comparison
EMOTIONS = ["neutral", "fear", "angry"]


def _load_pcm_16k_mono(raw_bytes: bytes) -> bytes:
    import io

    with wave.open(io.BytesIO(raw_bytes), "rb") as w:
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
    print("Downloading TESS (real human emotional speech, ~220MB, one-time)...")
    zip_path = hf_hub_download("myleslinder/tess", "data/tess.zip", repo_type="dataset")
    zf = zipfile.ZipFile(zip_path)

    # Build {word: {emotion: zip_entry_name}} for this actor, keep only
    # words present in ALL three emotions for a fair matched comparison.
    by_word: dict[str, dict[str, str]] = {}
    for name in zf.namelist():
        base = Path(name).stem  # e.g. "OAF_fat_disgust"
        parts = base.split("_")
        if len(parts) != 3 or parts[0] != ACTOR:
            continue
        actor, word, emotion = parts
        by_word.setdefault(word, {})[emotion] = name

    words = sorted(w for w, emos in by_word.items() if all(e in emos for e in EMOTIONS))[:N_WORDS]
    print(f"Testing {len(words)} words x {len(EMOTIONS)} emotions = {len(words) * len(EMOTIONS)} real clips, actor={ACTOR}")

    print("Loading real VAD + FasterWhisperSTT...")
    vad = VAD()
    stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    results: dict[str, list[bool]] = {e: [] for e in EMOTIONS}
    misses: dict[str, list[str]] = {e: [] for e in EMOTIONS}

    for word in words:
        for emotion in EMOTIONS:
            raw = zf.read(by_word[word][emotion])
            pcm = _load_pcm_16k_mono(raw)
            segments = vad.segment_speech(pcm)
            if not segments:
                results[emotion].append(False)
                misses[emotion].append(f"{word}: no speech detected")
                continue
            seg = max(segments, key=lambda s: s.end_ms - s.start_ms)
            transcript = stt.transcribe(seg.pcm).text.lower()
            correct = word.lower() in transcript
            results[emotion].append(correct)
            if not correct:
                misses[emotion].append(f"{word}: transcript={transcript!r}")

    print(f"\n{'=' * 70}\nSTT WORD-RECOGNITION ACCURACY BY EMOTION (real human speech, n={len(words)} each)\n{'=' * 70}")
    for emotion in EMOTIONS:
        acc = sum(results[emotion]) / len(results[emotion])
        print(f"{emotion:<10}{acc:>8.1%}")

    print("\nMisses by emotion:")
    for emotion in EMOTIONS:
        if misses[emotion]:
            print(f"  {emotion}:")
            for m in misses[emotion]:
                print(f"    {m}")


if __name__ == "__main__":
    main()
