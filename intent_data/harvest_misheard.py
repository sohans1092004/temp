"""Label-0 training text from REAL STT mistakes (Kaggle/Colab GPU). Run from the Model folder:
    python -m intent_data.harvest_misheard [--langs te_in,hi_in,...] [--max-clips 300]

Whisper small.en and Parakeet are English-only: given Telugu/Hindi/... speech they still
output English ("What's the name of the police?"). Those invented sentences were a large
share of the real false alarms (2026-09-27). This runs both live STTs over FLEURS non-English
speech (google/fleurs, CC-BY-4.0; the dev split, read speech) and keeps every non-empty
transcript -> intent_data/misheard/fleurs_<lang>.csv (text, source, stt, udk_hit, clip),
which build_v2.py merges as label 0. `udk_hit` = the UDK the exact/fuzzy rules would raise on
that text (no semantic model loaded, so it under-counts) -- the hardest negatives.

FLEURS is read speech from other speakers: the test audio (Andhra Pradesh YouTube clips,
the Telugu recordings) never enters training. Resumable: a language already written is skipped.
"""
from __future__ import annotations

import csv
import io
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "misheard"
LANGS = ["te_in", "hi_in", "kn_in", "ta_in", "ml_in", "mr_in", "bn_in"]
SR = 16_000


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def clips(lang: str, max_clips: int):
    """(name, 16 kHz int16 PCM bytes) from FLEURS <lang> dev split."""
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    tar = hf_hub_download("google/fleurs", f"data/{lang}/audio/dev.tar.gz", repo_type="dataset")
    n = 0
    with tarfile.open(tar) as t:
        for m in t:
            if n >= max_clips or not m.name.endswith(".wav"):
                continue
            audio, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            yield Path(m.name).name, (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            n += 1


def udk_hit(text: str, udks) -> str:
    from udk_engine import TRIGGER_VERIFY_THRESHOLD, match_transcript

    m = match_transcript(text, udks)
    return m.udk.udk_id if m.udk is not None and m.confidence >= TRIGGER_VERIFY_THRESHOLD else ""


def main() -> None:
    import torch
    from stt import FasterWhisperSTT, ParakeetSTT
    from udks import general_udks

    langs = arg("--langs", ",".join(LANGS)).split(",")
    max_clips = arg("--max-clips", 300)
    OUT.mkdir(exist_ok=True)
    todo = [l for l in langs if not (OUT / f"fleurs_{l}.csv").exists()]
    if not todo:
        print("all done")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stts = {"whisper": FasterWhisperSTT(model_size="small.en", device=dev, compute_type="float16" if dev == "cuda" else "int8"),
            "parakeet": ParakeetSTT(device=dev)}
    udks = general_udks("en")
    for lang in todo:
        rows = []
        for i, (name, pcm) in enumerate(clips(lang, max_clips)):
            for stt_name, stt in stts.items():
                text = stt.transcribe(pcm).text.strip()
                if text:
                    rows.append({"text": text, "source": f"fleurs_{lang}", "stt": stt_name, "udk_hit": udk_hit(text, udks), "clip": name})
            if i % 50 == 0:
                print(f"  {lang} {i + 1}/{max_clips}", flush=True)
        tmp = OUT / f"fleurs_{lang}.csv.part"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text", "source", "stt", "udk_hit", "clip"])
            w.writeheader()
            w.writerows(rows)
        tmp.replace(OUT / f"fleurs_{lang}.csv")  # only a finished language counts as done
        hits = sum(bool(r["udk_hit"]) for r in rows)
        print(f"{lang}: {len(rows)} transcripts, {hits} would raise a UDK today", flush=True)


if __name__ == "__main__":
    main()
