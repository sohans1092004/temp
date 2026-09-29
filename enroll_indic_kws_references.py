"""
Same one-time setup as enroll_kws_references.py, for the four non-English
languages -- generates reference clips per general UDK phrase (via
MMS-TTS instead of Windows SAPI, since it's cross-platform and already
used throughout this project's multilingual work) and saves a persisted
embedding bank per language via Wav2Vec2DTWSpotter.save_references().

Real finding behind INDIC_KWS_MODELS/INDIC_KWS_THRESHOLDS (kws.py): a
raw self-supervised multilingual model (facebook/wav2vec2-large-xlsr-53)
was tried first and produced ZERO usable signal -- 0/10 real paraphrases
correctly matched their own enrolled phrase as closest, positive and
negative distances completely overlapping (both in the 0.000-0.025
range). Root cause: this class's DTW approach needs a model that's been
fine-tuned for ASR (which sharpens word/phrase-level structure in the
hidden states), not just self-supervised pretrained -- confirmed by the
fact English's own KWS model (facebook/wav2vec2-base-960h) IS an
ASR-fine-tuned checkpoint, not a raw one. Switching to the Vakyansh
project's per-language ASR-fine-tuned wav2vec2 models produced a real,
usable signal for all four languages (7-8/10 correctly closest-matched),
though the calibration here is much lighter-weight than English's
(10 paraphrase positives + 20 negatives per language, vs. English's
120-clip corpus) -- treat these thresholds as a real starting point, not
a finished calibration.

Usage:
    python3 enroll_indic_kws_references.py
    python3 enroll_indic_kws_references.py --languages te hi
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from kws import INDIC_KWS_MODELS, SAMPLE_RATE, Wav2Vec2DTWSpotter
from udks import SUPPORTED_LANGUAGES, general_udks

TTS_REPOS = {
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "kn": "facebook/mms-tts-kan",
    "ta": "facebook/mms-tts-tam",
}


def _synth_pcm(tts_model, tts_tokenizer, text: str) -> bytes:
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    audio_f32 = output.squeeze().numpy()
    tts_sr = tts_model.config.sampling_rate
    duration = len(audio_f32) / tts_sr
    src_x = np.linspace(0, duration, num=len(audio_f32), endpoint=False)
    dst_x = np.linspace(0, duration, num=int(duration * SAMPLE_RATE), endpoint=False)
    resampled = np.interp(dst_x, src_x, audio_f32)
    return (np.clip(resampled, -1, 1) * 32767.0).astype("<i2").tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=[lang for lang in SUPPORTED_LANGUAGES if lang != "en"])
    parser.add_argument("--out-prefix", default="kws_references", help="output files are <prefix>_<lang>.npz")
    args = parser.parse_args()

    for lang in args.languages:
        print(f"\n=== {lang} ===")
        print(f"Loading MMS-TTS for {lang}...")
        tts_model = VitsModel.from_pretrained(TTS_REPOS[lang])
        tts_tokenizer = AutoTokenizer.from_pretrained(TTS_REPOS[lang])

        print(f"Loading {INDIC_KWS_MODELS[lang]}...")
        spotter = Wav2Vec2DTWSpotter(model_name=INDIC_KWS_MODELS[lang])

        udks = general_udks(lang)
        for udk in udks:
            print(f"  {udk.udk_id}")  # phrase text omitted -- non-Latin scripts crash Windows' cp1252 console
            spotter.enroll(udk.udk_id, _synth_pcm(tts_model, tts_tokenizer, udk.phrase))

        out_path = f"{args.out_prefix}_{lang}.npz"
        spotter.save_references(out_path)
        print(f"Saved {len(udks)} phrases to {out_path}")


if __name__ == "__main__":
    main()
