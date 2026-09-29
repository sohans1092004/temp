"""Fine-tune the semantic layer's own MiniLM with hard negatives, so "I'm not in danger"
lands FAR from "I'm in danger" in embedding space. Run from the Model folder (GPU):
    python -m intent_data.finetune_minilm

The semantic layer scores a transcript as cos(UDK phrase, transcript), so training uses
exactly that shape: pairs (UDK phrase, sentence, 1/0) with OnlineContrastiveLoss.
  - label 1 rows: paired with their own UDK (INTENT rows: with the nearest UDK)
  - label 0 rows: paired with the UDK the UNTUNED model finds nearest -- the hardest
    negative; negation pairs with their own UDK (the "not" is the only difference)
Protocol (same as nli_eval.py): epoch + threshold on DEV (recall >= 95%), TEST once per
category, prepped replay as REGRESSION ONLY. The untuned model is scored the same way
as the baseline. Best model -> intent_data/minilm_negft/ (a drop-in for semantic.py's
model_name).
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from nli_eval import CONDS, LOGS, regression, report, thr_for  # noqa: E402
from udks import GENERAL_UDKS  # noqa: E402

BASE = "paraphrase-multilingual-MiniLM-L12-v2"  # semantic.py DEFAULT_MODEL
EPOCHS, LR, BATCH, SEED = 4, 2e-5, 32, 0
OUT = HERE / "minilm_results.json"
SAVE = HERE / "minilm_negft"
PHRASES = {u.udk_id: u.phrase for u in GENERAL_UDKS}


class Scorer:
    """score(text) = max cosine to the 20 UDK phrases -- what the semantic layer does."""

    def __init__(self, model):
        self.model = model
        self.ids = list(PHRASES)
        self.ref = model.encode([PHRASES[u] for u in self.ids], normalize_embeddings=True)

    def sims(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.ids)))
        return self.model.encode(texts, normalize_embeddings=True, batch_size=128) @ self.ref.T

    def score(self, texts: list[str]) -> np.ndarray:
        s = self.sims(texts)
        return s.max(axis=1) if len(s) else np.zeros(0)


def pairs(train: list[dict], base: Scorer) -> list:
    from sentence_transformers import InputExample

    nearest = base.sims([r["text"] for r in train]).argmax(axis=1)
    out = []
    for r, k in zip(train, nearest):
        own = r["udk_id"] if r["udk_id"] in PHRASES else None
        if r["category"].startswith("minimal_pair_neg"):
            own = r["template_id"].split("/")[1]  # negpair/UDK_xx: pair with its own UDK
        udk = own or base.ids[k]
        out.append(InputExample(texts=[PHRASES[udk], r["text"]], label=float(r["label"])))
    return out


def evaluate(scorer: Scorer, rows: dict, replay) -> dict:
    p = {s: scorer.score([r["text"] for r in v]) for s, v in rows.items()}
    y = {s: np.array([int(r["label"]) for r in v]) for s, v in rows.items()}
    thr = thr_for(p["dev"], y["dev"])
    hit = p["test"] >= thr
    per = defaultdict(lambda: [0, 0])
    for r, h in zip(rows["test"], hit):
        c = r["category"] + (f" [{r['label']}]" if r["category"].startswith("minimal_pair") else "")
        per[c][0] += int(h)
        per[c][1] += 1
    res = {"threshold": thr, "dev_fpr": float((p["dev"][y["dev"] == 0] >= thr).mean()),
           "test_recall": float(hit[y["test"] == 1].mean()), "test_fpr": float(hit[y["test"] == 0].mean()),
           "per_category": dict(per)}
    if replay is not None:
        res["regression"] = regression(scorer, thr, replay)
    return res


def cpu_ms(path_or_name) -> float:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(str(path_or_name), device="cpu")
    ref = m.encode(list(PHRASES.values()), normalize_embeddings=True)
    t0 = time.perf_counter()
    for _ in range(40):
        m.encode(["someone is following me and I'm not okay"], normalize_embeddings=True) @ ref.T
    return round((time.perf_counter() - t0) / 40 * 1000, 1)


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = defaultdict(list)
    for r in csv.DictReader(open(HERE / "intent_dataset.csv", encoding="utf-8")):
        rows[r["split"]].append(r)
    evalrows = {s: rows[s] for s in ("dev", "test")}
    replay = None
    if all((LOGS / f"prepped_gpu_{c}.log").exists() for c in CONDS):
        import scratch_replay as R

        replay = {c: R.parse(str(LOGS / f"prepped_gpu_{c}.log")) for c in CONDS}

    model = SentenceTransformer(BASE, device=device)
    base = Scorer(model)
    u = "minilm/untuned"  # report() labels columns by the part after "/"
    results = {u: evaluate(base, evalrows, replay)}
    results[u]["cpu_ms_per_window"] = cpu_ms(BASE)
    print(f"untuned: dev thr {results[u]['threshold']:.2f}, dev FPR {results[u]['dev_fpr']:.1%}", flush=True)

    train = pairs(rows["train"], base)
    print(f"{len(train)} training pairs ({sum(e.label == 1 for e in train)} positive)", flush=True)
    loader = DataLoader(train, shuffle=True, batch_size=BATCH)
    loss = losses.OnlineContrastiveLoss(model)
    best = None
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.fit(train_objectives=[(loader, loss)], epochs=1, optimizer_params={"lr": LR},
                  warmup_steps=len(loader) // 10 if ep == 1 else 0, show_progress_bar=False)
        s = Scorer(model)
        p = s.score([r["text"] for r in evalrows["dev"]])
        y = np.array([int(r["label"]) for r in evalrows["dev"]])
        thr = thr_for(p, y)
        fpr = float((p[y == 0] >= thr).mean())
        print(f"epoch {ep}: dev thr {thr:.2f}, dev FPR at 95% recall {fpr:.1%} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if best is None or fpr < best:  # epoch chosen on DEV only
            best = fpr
            model.save(str(SAVE))
    tuned = SentenceTransformer(str(SAVE), device=device)
    name = "minilm/negation-finetuned"
    results[name] = evaluate(Scorer(tuned), evalrows, replay)
    results[name]["cpu_ms_per_window"] = cpu_ms(SAVE)
    OUT.write_text(json.dumps(results, indent=1))
    for n, r in results.items():
        print(f"{n}: thr {r['threshold']:.2f} | TEST recall {r['test_recall']:.1%}, FPR {r['test_fpr']:.1%} | "
              f"CPU {r['cpu_ms_per_window']} ms", flush=True)
    report(results)
    print(f"\nsaved: {SAVE}  (use as SentenceTransformerSemanticMatcher(model_name=...) with the threshold above)")


if __name__ == "__main__":
    main()
