"""Trace the semantic layer's false alarms (English, MiniLM @ 0.55) and
compare it against LaBSE on the same text, so any threshold/model change
is picked from a measured separation, not a guess.

Positives (a real paraphrase should fire):
  - calibrate_kws_threshold.PARAPHRASES: 1 natural paraphrase per UDK (20)
  - real_recordings/mixed manifest phrases that are NOT exact UDKs (12)
Negatives (nothing should fire):
  - evaluate_pipeline_corpus.NEGATIVES: ordinary sentences (20)
  - real Whisper tiny.en transcripts of every VAD segment in
    real_recordings/negatives (real ordinary speech, 23 files)
  - Model-snr's ADVERSARIAL_SENTENCES (UDK words inside harmless
    sentences; most contain the exact phrase, so the exact layer fires on
    them regardless -- reported separately, semantic-only view)

Scored exactly as udk_engine.match_transcript does: best similarity over
the 20 UDKs, both sides passed through _normalize().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(str(Path(__file__).parent.parent / "Model-snr"))

from calibrate_kws_threshold import PARAPHRASES
from evaluate_full_system import load_real_pcm
from evaluate_pipeline_corpus import NEGATIVES
from udk_engine import _normalize
from udks import GENERAL_UDKS
from validate_forced_scoring import ADVERSARIAL_SENTENCES

REAL = Path(__file__).parent / "real_recordings"


def real_negative_transcripts() -> list[str]:
    from stt import FasterWhisperSTT
    from vad import VAD

    vad, stt, out = VAD(), FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8"), []
    for p in sorted((REAL / "negatives").iterdir()):
        if p.suffix.lower() in (".mp3", ".wav", ".m4a"):
            out += [t for seg in vad.segment_speech(load_real_pcm(p)) if (t := stt.transcribe(seg.pcm).text.strip())]
    return out


def main() -> None:
    import json

    manifest = json.loads((REAL / "mixed" / "manifest.json").read_text(encoding="utf-8"))
    exact = {u.phrase.lower() for u in GENERAL_UDKS}
    positives = [("kws_paraphrase", t) for t in PARAPHRASES.values()]
    positives += [("mixed_paraphrase", e["phrase"]) for e in manifest if e["phrase"].lower() not in exact]
    print("Transcribing real negatives (VAD + tiny.en)...")
    negatives = [("ordinary", t) for t in NEGATIVES] + [("real_negative_seg", t) for t in real_negative_transcripts()]
    adversarial = [("adversarial", t) for t in ADVERSARIAL_SENTENCES.values()]
    print(f"  positives={len(positives)} negatives={len(negatives)} adversarial={len(adversarial)}")

    from sentence_transformers import SentenceTransformer, util

    phrases = [_normalize(u.phrase) for u in GENERAL_UDKS]
    for model_name in ("paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/LaBSE"):
        model = SentenceTransformer(model_name)
        pe = model.encode(phrases, convert_to_tensor=True)

        def best(text: str) -> tuple[float, str]:
            sims = util.cos_sim(model.encode(_normalize(text), convert_to_tensor=True), pe)[0]
            i = int(sims.argmax())
            return float(sims[i]), GENERAL_UDKS[i].udk_id

        P = [(k, t, *best(t)) for k, t in positives]
        N = [(k, t, *best(t)) for k, t in negatives]
        A = [(k, t, *best(t)) for k, t in adversarial]
        print(f"\n{'=' * 90}\n{model_name}\n{'=' * 90}")
        print("Top-scoring NEGATIVES (would-be false alarms):")
        for k, t, s, u in sorted(N, key=lambda r: -r[2])[:15]:
            print(f"  {s:.3f} {u} [{k}] {t[:90]!r}")
        print("Lowest-scoring POSITIVES (would-be misses):")
        for k, t, s, u in sorted(P, key=lambda r: r[2])[:12]:
            print(f"  {s:.3f} {u} [{k}] {t!r}")
        print(f"\n  {'thr':>5} {'recall':>10} {'FP ordinary':>12} {'FP real segs':>13} {'FP adversarial':>15}")
        lo = min(r[2] for r in P + N)
        for thr in [round(x * 0.05, 2) for x in range(int(lo * 20), 20)]:
            rec = sum(r[2] >= thr for r in P)
            fo = sum(r[2] >= thr for r in N if r[0] == "ordinary")
            fr = sum(r[2] >= thr for r in N if r[0] == "real_negative_seg")
            fa = sum(r[2] >= thr for r in A)
            nr = sum(r[0] == "real_negative_seg" for r in N)
            print(f"  {thr:>5.2f} {rec:>4}/{len(P):<5} {fo:>6}/{len(NEGATIVES):<5} {fr:>6}/{nr:<6} {fa:>8}/{len(A):<6}")


if __name__ == "__main__":
    main()
