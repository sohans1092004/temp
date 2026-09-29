"""Zero-shot NLI models as a UDK-intent signal (text only, no training).
Premise = transcript, hypothesis = "the speaker is in danger / asking for help";
score = P(entailment). Run from the Model folder:
    python -m intent_data.nli_eval [model ...]

Protocol (same as train_classifier.py, fixed in advance):
  - threshold per model chosen on DEV: highest keeping dev recall >= 95%
  - scored once on TEST (unseen templates), per category
  - REGRESSION ONLY: replayed prepped transcripts -- model alone, and as a gate on WEAK
    alerts (non-exact, conf < 0.75). Nothing tuned on prepped.
  - latency: CPU, batch of 1, per window
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HYPOTHESIS = "The speaker is in immediate danger or is asking for help."
MODELS = [
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-v3-small",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
]
DEV_RECALL = 0.95
WEAK_CONF = 0.75
LOGS = next((p for p in (HERE / "logs", Path.home() / "Downloads") if (p / "prepped_gpu_normal.log").exists()), HERE / "logs")  # bundled logs first (Colab)
CONDS = ["normal", "quiet", "loud", "crowd", "pocket", "phone"]
OUT = HERE / "nli_results.json"


class NLIScorer:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        # entailment index from the model's own config -- label order differs between models
        self.ent = next(i for l, i in labels.items() if l.startswith("entail"))
        print(f"  {name}: labels {self.model.config.id2label} -> entailment = {self.ent}", flush=True)

    def score(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                t = texts[i:i + batch]
                enc = self.tok(t, [HYPOTHESIS] * len(t), truncation=True, max_length=256, padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.ent].numpy())
        return np.concatenate(out) if out else np.zeros(0)


def thr_for(p, y):
    for t in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if (p[y == 1] >= t).mean() >= DEV_RECALL:
            return float(t)
    return 0.01


def evaluate(name: str, rows: dict, replay) -> dict:
    m = NLIScorer(name)
    p = {s: m.score([r["text"] for r in v]) for s, v in rows.items()}
    y = {s: np.array([int(r["label"]) for r in v]) for s, v in rows.items()}
    thr = thr_for(p["dev"], y["dev"])
    hit = p["test"] >= thr
    per = defaultdict(lambda: [0, 0])
    for r, h in zip(rows["test"], hit):
        c = r["category"] + (f" [{r['label']}]" if r["category"].startswith("minimal_pair") else "")
        per[c][0] += int(h)
        per[c][1] += 1
    # latency: one window at a time, CPU
    sample = [r["text"] for r in rows["test"][:40]]
    t0 = time.perf_counter()
    for s in sample:
        m.score([s], batch=1)
    ms = (time.perf_counter() - t0) / len(sample) * 1000
    res = {"threshold": thr, "test_recall": float(hit[y["test"] == 1].mean()), "test_fpr": float(hit[y["test"] == 0].mean()),
           "per_category": {c: v for c, v in per.items()}, "cpu_ms_per_window": round(ms, 1)}
    if replay is not None:
        res["regression"] = regression(m, thr, replay)
    return res


def regression(m, thr, data) -> dict:
    """Prepped transcripts replayed through today's rules (scratch_replay), compared with:
    NLI alone (any segment >= thr) and rules whose WEAK alerts must also pass NLI."""
    import scratch_replay as R

    t = {"rules": [0, 0, 0], "nli_alone": [0, 0, 0], "rules+gate": [0, 0, 0]}  # danger caught, controls alerting, n
    lost, silenced = [], []
    cache = {}

    def s(texts):
        need = [x for x in texts if x not in cache]
        for x, v in zip(need, m.score(need)):
            cache[x] = float(v)
        return [cache[x] for x in texts]

    for c, clips in data.items():
        for name, clip in clips.items():
            _, events = R.replay(clip)
            segs = [x for xs in clip["paths"].values() for x in xs]
            rules = bool(events)
            alone = bool(segs) and max(s(segs)) >= thr
            kept = [e for _, e in events if e.layer.startswith("exact") or e.confidence >= WEAK_CONF or s([e.transcript])[0] >= thr]
            gated = bool(kept)
            for k, v in (("rules", rules), ("nli_alone", alone), ("rules+gate", gated)):
                t[k][0 if clip["danger"] else 1] += v
                t[k][2] += 0
            if clip["danger"] and rules and not gated:
                lost.append(f"{c}/{name[:45]}")
            if not clip["danger"] and rules and not gated:
                silenced.append(f"{c}/{name[:45]}")
    n_d = sum(clip["danger"] for clips in data.values() for clip in clips.values())
    n_c = sum(not clip["danger"] for clips in data.values() for clip in clips.values())
    return {"danger_n": n_d, "control_n": n_c, **{k: v[:2] for k, v in t.items()}, "gate_lost": lost, "gate_silenced": silenced}


def main() -> None:
    names = sys.argv[1:] or MODELS
    rows = defaultdict(list)
    for r in csv.DictReader(open(HERE / "intent_dataset.csv", encoding="utf-8")):
        rows[r["split"]].append(r)
    rows = {s: rows[s] for s in ("dev", "test")}
    replay = None
    if all((LOGS / f"prepped_gpu_{c}.log").exists() for c in CONDS):
        import scratch_replay as R

        replay = {c: R.parse(str(LOGS / f"prepped_gpu_{c}.log")) for c in CONDS}
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n in names:
        if n in results:
            print(f"[skip] {n} (done)", flush=True)
            continue
        print(f"\n== {n}", flush=True)
        t0 = time.time()
        results[n] = evaluate(n, rows, replay)
        results[n]["eval_minutes"] = round((time.time() - t0) / 60, 1)
        OUT.write_text(json.dumps(results, indent=1))
        r = results[n]
        print(f"  dev threshold {r['threshold']:.2f} | TEST recall {r['test_recall']:.1%}, FPR {r['test_fpr']:.1%} | "
              f"CPU {r['cpu_ms_per_window']} ms/window", flush=True)
    report(results)


def report(results: dict) -> None:
    cats = ["paraphrase", "paraphrase+context", "paraphrase+asr_heavy", "minimal_pair [1]", "minimal_pair [0]", "minimal_pair+delayed [1]", "minimal_pair+delayed [0]",
            "minimal_pair_neg [1]", "minimal_pair_neg [0]", "minimal_pair_neg+context [0]", "negation_danger", "negation_danger+context",
            "negation", "negation+context", "quote", "hyperbole", "instruction", "near_miss", "own_speech_nonthreat",
            "retraction", "chatter+context", "hinglish"]
    print("\nTEST split (unseen templates): % flagged as distress -- positives should be HIGH, others LOW")
    print(f"{'category':<24}" + "".join(f"{n.split('/')[1][:18]:>20}" for n in results))
    for c in cats:
        cells = []
        for r in results.values():
            k, n = r["per_category"].get(c, (0, 0))
            cells.append(f"{k}/{n} ({k / n:.0%})" if n else "-")
        print(f"{c:<24}" + "".join(f"{x:>20}" for x in cells))
    print("(minimal_pair [1] = danger side: should be HIGH; [0] = harmless side: should be LOW)")
    for n, r in results.items():
        g = r.get("regression")
        if g:
            print(f"\nREGRESSION {n}: danger {g['danger_n']}, controls {g['control_n']}")
            for k in ("rules", "nli_alone", "rules+gate"):
                print(f"   {k:<12} danger caught {g[k][0]}/{g['danger_n']}   controls alerting {g[k][1]}/{g['control_n']}")
            print(f"   gate LOST {len(g['gate_lost'])}: {g['gate_lost'][:6]}")
            print(f"   gate silenced {len(g['gate_silenced'])}: {sorted(set(x.split('/')[1] for x in g['gate_silenced']))}")


if __name__ == "__main__":
    main()
