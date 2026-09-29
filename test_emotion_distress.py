"""STANDALONE TEST (2026-09-23) -- nothing here is wired into the pipeline.
Question: can a categorical speech-emotion model (angry / fearful / sad /
happy / neutral ...) plus BEATs tell real distress apart from annoyed or
playful speech well enough to BOOST a UDK match's confidence
(TRIGGER_VERIFY -> TRIGGER_ALL)? Boost-only by design: a UDK is a code
phrase a user may say calmly on purpose, so emotion must never veto.

Unlike test_ser_arousal_scoring.py (audeering arousal, a DEAD END: 56% of
real ordinary segments scored "high arousal"), these models name the
emotion, so fear can in principle be told apart from excitement/annoyance.

Test sets (both unseen by these models; RAVDESS is skipped because both
were likely trained on it):
  - prepped_data/ (17 clips, held-out TEST ONLY): does "Real fear" /
    "Panicked" score fearful/sad while "Annoyed" / "Playful" / calm don't?
  - real_recordings/negatives (every VAD segment): how often does ordinary
    speech score as distress? = the false-boost rate.

Distress score = P(fearful) + P(sad) (crying/pleading). Angry is reported
separately: an aggressor's voice and an annoyed friend both sound angry.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from evaluate_prepped_data import DATA as PREPPED, expected_udks, load_pcm
from evaluate_full_system import REAL_RECORDINGS, load_real_pcm

MODELS = [
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
]
FEAR, SAD, ANGRY = ("fear", "fearful", "fea"), ("sad",), ("angry", "anger", "ang")


def load_classifier(model_id: str):
    """-> score(pcm) -> {label: prob}. Not transformers.pipeline: it imports
    torchcodec, broken in this environment (same issue as overlap_gate.py).
    Raises if any classifier weight is missing -- a silently random head
    would produce confident-looking garbage."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoFeatureExtractor

    cfg = AutoConfig.from_pretrained(model_id)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    state = load_file(hf_hub_download(model_id, "model.safetensors"))

    if any(k.startswith("classifier.dense.") for k in state):
        # ehcalabres: custom head from its training script (mean-pool ->
        # dense -> tanh -> output), unknown to Wav2Vec2ForSequenceClassification.
        from transformers import Wav2Vec2Model

        enc = Wav2Vec2Model(cfg)
        missing, _ = enc.load_state_dict({k[len("wav2vec2."):]: v for k, v in state.items() if k.startswith("wav2vec2.")}, strict=False)
        assert not [m for m in missing if "masked_spec_embed" not in m], f"encoder weights missing: {missing}"
        dense = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)
        out = torch.nn.Linear(cfg.hidden_size, len(labels))
        dense.load_state_dict({"weight": state["classifier.dense.weight"], "bias": state["classifier.dense.bias"]})
        out.load_state_dict({"weight": state["classifier.output.weight"], "bias": state["classifier.output.bias"]})
        enc.eval()

        def logits(x):
            return out(torch.tanh(dense(enc(**x).last_hidden_state.mean(dim=1))))
    else:
        from transformers import AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(model_id, output_loading_info=False)
        missing, _ = model.load_state_dict(state, strict=False)
        assert not [m for m in missing if "classifier" in m or "projector" in m], f"classifier weights missing: {missing}"
        model.eval()

        def logits(x):
            return model(**x).logits

    def score(pcm: bytes) -> dict[str, float]:
        x = fe(pcm_to_f32(pcm), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(logits(x), dim=-1)[0]
        return dict(zip(labels, p.tolist()))

    return score


def pcm_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def group(scores: dict[str, float]) -> tuple[float, float, str]:
    """-> (distress = fear + sad, angry, top label)"""
    def p(names):
        return sum(v for k, v in scores.items() if k.lower() in names)
    return p(FEAR) + p(SAD), p(ANGRY), max(scores, key=scores.get)


def prepped_outcomes() -> dict[str, str]:
    """Per-file strongest decision from the small.en + per-sentence run."""
    log = Path(__file__).parent / "prepped_data_eval_output_sentences_small.en.log"
    out, cur = {}, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and not line.startswith((" ", "=", "SUMMARY")) and not line.startswith(("Loading", "C:\\", "[")):
            cur = line.strip()
            out.setdefault(cur, "nothing")
        m = re.search(r"-> (TRIGGER_ALL|TRIGGER_VERIFY)", line)
        if cur and m and (out[cur] != "TRIGGER_ALL"):
            out[cur] = m.group(1)
    return out


def main() -> None:


    from beats_distress_detector import BEATsDistressDetector
    from vad import VAD

    beats, vad = BEATsDistressDetector(), VAD()
    prepped = [(p.stem, load_pcm(p), bool(expected_udks(p.name))) for p in sorted(PREPPED.glob("*.mp3"))]
    neg_segments = []
    for p in sorted((REAL_RECORDINGS / "negatives").iterdir()):
        if p.suffix.lower() in (".mp3", ".wav", ".m4a"):
            neg_segments += [seg.pcm for seg in vad.segment_speech(load_real_pcm(p))]
    print(f"prepped clips={len(prepped)}  real-negative segments={len(neg_segments)}")

    beats_prepped = {name: beats.score(pcm) for name, pcm, _ in prepped}
    beats_neg = [beats.score(s) for s in neg_segments]
    outcomes = prepped_outcomes()

    for model_id in MODELS:
        print(f"\n{'=' * 110}\n{model_id}\n{'=' * 110}")
        score = load_classifier(model_id)

        print(f"  {'clip':<62} {'danger?':<8} {'top':<10} {'distress':>9} {'angry':>7} {'BEATs':>7}  outcome now")
        rows = []
        for name, pcm, danger in prepped:
            d, a, top = group(score(pcm))
            rows.append((name, danger, d, a, top))
            print(f"  {name[:62]:<62} {'YES' if danger else 'no':<8} {top:<10} {d:>9.2f} {a:>7.2f} {beats_prepped[name]:>7.3f}  {outcomes.get(name, '?')}")

        neg = [group(score(s)) for s in neg_segments]
        print(f"\n  Real ordinary speech ({len(neg)} segments) -- how often would it BOOST (false-boost rate):")
        for t in (0.3, 0.5, 0.7, 0.9):
            fb = sum(d >= t for d, _, _ in neg)
            fa = sum(a >= t for _, a, _ in neg)
            dang = [r for r in rows if r[1]]
            safe = [r for r in rows if not r[1]]
            print(f"    distress >= {t:.1f}: real-negative segs {fb:>3}/{len(neg)} ({100 * fb / len(neg):4.1f}%) | "
                  f"prepped danger {sum(r[2] >= t for r in dang)}/{len(dang)}, prepped controls {sum(r[2] >= t for r in safe)}/{len(safe)}"
                  f"   || angry >= {t:.1f}: real-neg {fa}/{len(neg)}")
        tops = {}
        for _, _, top in neg:
            tops[top] = tops.get(top, 0) + 1
        print(f"  top emotion on real ordinary speech: {dict(sorted(tops.items(), key=lambda kv: -kv[1]))}")

    print(f"\n{'=' * 110}\nBEATs (already live, corroboration-only, threshold 0.30)\n{'=' * 110}")
    for t in (0.05, 0.10, 0.30):
        print(f"  BEATs >= {t:.2f}: real-negative segs {sum(b >= t for b in beats_neg)}/{len(beats_neg)} | "
              f"prepped danger {sum(beats_prepped[n] >= t for n, _, dg in prepped if dg)}/{sum(dg for *_, dg in prepped)}, "
              f"controls {sum(beats_prepped[n] >= t for n, _, dg in prepped if not dg)}/{sum(not dg for *_, dg in prepped)}")


if __name__ == "__main__":
    main()
