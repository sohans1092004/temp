"""
Calibrates kws.py's INDIC_KWS_THRESHOLDS against a real, larger measured
corpus per language -- replacing the original "10 paraphrase positives +
20 negatives" starting point (kws.py's own docstring already flagged that
as "a real starting point, not a finished calibration").

Same methodology as calibrate_kws_threshold.py (paraphrase-only positives,
so testing never self-matches an enrolled reference's exact words), scaled
up to ALL 20 UDKs per language (not 10) and reusing
calibrate_kws_threshold.py's own English PARAPHRASES/NEGATIVES content,
machine-translated via NLLB -- not fresh guesses, the same real,
already-used-throughout-this-project sentences.

Real constraint this project's English calibration doesn't have: MMS-TTS
(facebook/mms-tts-*) ships ONE voice per language, not several like
Windows SAPI -- there is no "hold out a voice" option here. Substitute:
this session's own real finding that MMS-TTS's duration predictor/flow is
stochastic (re-synthesizing "the same" text produces audibly different
audio each call -- see README.md's degraded-condition noise-floor
discussion) is used deliberately here as a stand-in for voice diversity --
each phrase is synthesized TWICE, independently, giving real acoustic
(not just lexical) variation between the two "voices" per language. Final
corpus per language: 20 UDKs x 2 syntheses = 40 positives, 20 negatives x
2 syntheses = 40 negatives -- 80 real clips, the same scale as English's
current 80-clip corpus (calibrate_kws_threshold.py).
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

import kws as kws_module
from calibrate_kws_threshold import NEGATIVES, PARAPHRASES
from kws import INDIC_KWS_MODELS, SAMPLE_RATE, Wav2Vec2DTWSpotter
from udks import general_udks

TTS_REPOS = {"hi": "facebook/mms-tts-hin", "te": "facebook/mms-tts-tel", "kn": "facebook/mms-tts-kan", "ta": "facebook/mms-tts-tam"}
NLLB_CODES = {"hi": "hin_Deva", "te": "tel_Telu", "kn": "kan_Knda", "ta": "tam_Taml"}
N_SYNTH_PASSES = 2  # stand-in for "voice diversity" -- see module docstring
THRESHOLDS_TO_SWEEP = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]


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
    languages = sys.argv[1:] or ["te", "kn", "hi", "ta"]

    print("Loading NLLB translator...")
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    def translate(text: str, tgt_code: str) -> str:
        nllb_tokenizer.src_lang = "eng_Latn"
        inputs = nllb_tokenizer(text, return_tensors="pt")
        tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_code)
        out = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=64)
        return nllb_tokenizer.decode(out[0], skip_special_tokens=True)

    summary = {}
    for lang in languages:
        print(f"\n{'='*70}\n{lang}\n{'='*70}")
        udks = general_udks(lang)

        print(f"Loading MMS-TTS for {lang}...")
        tts_model = VitsModel.from_pretrained(TTS_REPOS[lang])
        tts_tokenizer = AutoTokenizer.from_pretrained(TTS_REPOS[lang])

        print(f"Loading {INDIC_KWS_MODELS[lang]} + real enrolled reference bank...")
        spotter = Wav2Vec2DTWSpotter(model_name=INDIC_KWS_MODELS[lang])
        spotter.load_references(f"kws_references_{lang}.npz")

        def distances_to_all(pcm: bytes) -> dict[str, float]:
            query = spotter._frame_embeddings(pcm)
            return {pid: min(kws_module._dtw_distance(query, ref) for ref in refs) for pid, refs in spotter._references.items()}

        print("Translating paraphrases + negatives via NLLB...")
        translated_paraphrases = {udk_id: translate(text, NLLB_CODES[lang]) for udk_id, text in PARAPHRASES.items()}
        translated_negatives = [translate(text, NLLB_CODES[lang]) for text in NEGATIVES]

        positives: list[tuple[str, float]] = []
        positive_misses: list[str] = []
        negatives: list[float] = []

        print(f"Synthesizing {len(udks) * N_SYNTH_PASSES} positive clips ({N_SYNTH_PASSES} passes/phrase)...")
        for udk in udks:
            paraphrase = translated_paraphrases[udk.udk_id]
            for _pass in range(N_SYNTH_PASSES):
                pcm = _synth_pcm(tts_model, tts_tokenizer, paraphrase)
                dists = distances_to_all(pcm)
                closest_id = min(dists, key=dists.get)
                if closest_id == udk.udk_id:
                    positives.append((udk.udk_id, dists[udk.udk_id]))
                else:
                    positive_misses.append(f"{udk.udk_id} confused with {closest_id} (dist {dists[closest_id]:.3f} vs own {dists[udk.udk_id]:.3f})")

        print(f"Synthesizing {len(translated_negatives) * N_SYNTH_PASSES} negative clips...")
        for text in translated_negatives:
            for _pass in range(N_SYNTH_PASSES):
                pcm = _synth_pcm(tts_model, tts_tokenizer, text)
                dists = distances_to_all(pcm)
                negatives.append(min(dists.values()))

        print(f"\n{len(positives)}/{len(udks) * N_SYNTH_PASSES} positives correctly closest-matched, {len(positive_misses)} confused with a different real UDK:")
        for m in positive_misses:
            print(f"  {m}")

        print(f"\n{'threshold':>10} {'recall':>10} {'false_pos_rate':>16}")
        lang_summary = {}
        for threshold in THRESHOLDS_TO_SWEEP:
            recall = sum(1 for _, d in positives if d <= threshold) / len(positives) if positives else 0.0
            fpr = sum(1 for d in negatives if d <= threshold) / len(negatives) if negatives else 0.0
            print(f"{threshold:>10.2f} {recall:>10.1%} {fpr:>16.1%}")
            lang_summary[threshold] = (recall, fpr)
        summary[lang] = lang_summary

        print(f"\nRaw positive distances (own-UDK, correctly-closest only): {sorted(round(d, 3) for _, d in positives)}")
        print(f"Raw negative (closest-any) distances: {sorted(round(d, 3) for d in negatives)}")

    print(f"\n\n{'='*70}\nSUMMARY (recall% / FPR% at each threshold)\n{'='*70}")
    header = "lang  " + "  ".join(f"{th:.2f}" for th in THRESHOLDS_TO_SWEEP)
    print(header)
    for lang in languages:
        row = "  ".join(f"{summary[lang][th][0]*100:3.0f}/{summary[lang][th][1]*100:3.0f}" for th in THRESHOLDS_TO_SWEEP)
        print(f"{lang:4s}  {row}")


if __name__ == "__main__":
    main()
