"""UDK-intent classifier: MiniLM sentence embeddings (the semantic layer's model) +
logistic regression. Text only, CPU, seconds. Run from the Model folder:
    python -m intent_data.train_classifier

Protocol (fixed in advance):
  - C and the threshold are chosen on DEV only: highest threshold with dev recall >= 95%.
  - Scored once on the generated TEST split (unseen templates), per category.
  - REGRESSION ONLY: replayed prepped_data transcripts (6 conditions), comparing
    rules alone / classifier alone / rules whose WEAK alerts must pass the classifier.
    Nothing is tuned on prepped_data.
"""
from __future__ import annotations

import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL = HERE / "intent_clf.pkl"
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEV_RECALL = 0.95
WEAK_CONF = 0.75  # same definition of a "weak" alert as distress_detector.combine
LOGS = next((p for p in (HERE / "logs", Path.home() / "Downloads") if (p / "prepped_gpu_normal.log").exists()), HERE / "logs")  # bundled logs first (Colab)
CONDS = ["normal", "quiet", "loud", "crowd", "pocket", "phone"]

sys.path.insert(0, str(HERE.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_enc = None


def embed(texts: list[str]) -> np.ndarray:
    global _enc
    if _enc is None:
        from sentence_transformers import SentenceTransformer

        _enc = SentenceTransformer(EMB_MODEL, device="cpu")
    return _enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


class IntentClassifier:
    def __init__(self, clf, thr: float):
        self.clf, self.thr = clf, thr

    def score(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(embed(texts))[:, 1] if texts else np.zeros(0)

    @staticmethod
    def load(path: Path = MODEL) -> "IntentClassifier":
        return pickle.load(open(path, "rb"))


def train() -> IntentClassifier:
    from sklearn.linear_model import LogisticRegression

    rows = list(csv.DictReader(open(HERE / "intent_dataset.csv", encoding="utf-8")))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    X = {s: embed([r["text"] for r in v]) for s, v in split.items()}
    y = {s: np.array([int(r["label"]) for r in v]) for s, v in split.items()}

    def thr_for(p, yy):  # highest threshold keeping recall >= DEV_RECALL
        for t in np.round(np.arange(0.99, 0.0, -0.01), 2):
            if (p[yy == 1] >= t).mean() >= DEV_RECALL:
                return float(t)
        return 0.01

    best = None
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X["train"], y["train"])
        p = clf.predict_proba(X["dev"])[:, 1]
        t = thr_for(p, y["dev"])
        fpr = float((p[y["dev"] == 0] >= t).mean())
        print(f"  C={C:<5} dev: thr {t:.2f}, recall {(p[y['dev'] == 1] >= t).mean():.1%}, false-positive rate {fpr:.1%}")
        if best is None or fpr < best[2]:
            best = (clf, t, fpr, C)
    clf, thr, _, C = best
    model = IntentClassifier(clf, thr)
    pickle.dump(model, open(MODEL, "wb"))
    print(f"chosen on DEV: C={C}, threshold {thr:.2f}\n")

    p = clf.predict_proba(X["test"])[:, 1]
    hit = p >= thr
    print("TEST split (unseen templates):")
    print(f"  recall {hit[y['test'] == 1].mean():.1%} ({hit[y['test'] == 1].sum()}/{(y['test'] == 1).sum()}), "
          f"false-positive rate {hit[y['test'] == 0].mean():.1%} ({hit[y['test'] == 0].sum()}/{(y['test'] == 0).sum()})")
    per = defaultdict(lambda: [0, 0])
    for r, h in zip(split["test"], hit):
        c = r["category"].split("+")[0] + ("+context" if r["category"].endswith("context") else "")
        per[c][0] += int(h)
        per[c][1] += 1
    for c, (k, n) in sorted(per.items()):
        kind = "caught" if c.startswith(("exact", "paraphrase")) else "flagged (false positive)"
        print(f"    {c:<26} {kind:<26} {k}/{n} ({k / n:.0%})")
    errs = [(float(pp), r["text"], r["label"]) for r, pp, h in zip(split["test"], p, hit) if h != (r["label"] == "1")]
    print("  sample test errors:")
    for pp, t, l in sorted(errs, key=lambda e: -abs(e[0] - thr))[:12]:
        print(f"    {'MISSED ' if l == '1' else 'FALSE +'} {pp:.2f}  {t[:90]}")
    return model


def regression(model: IntentClassifier) -> None:
    """Replayed prepped transcripts -- regression check only, nothing tuned here."""
    sys.path.insert(0, str(HERE))
    import scratch_replay as R  # the log parser + rule replay (copied into this folder)

    data = {c: R.parse(str(LOGS / f"prepped_gpu_{c}.log")) for c in CONDS if (LOGS / f"prepped_gpu_{c}.log").exists()}
    if not data:
        print("\n(no prepped_gpu_*.log in Downloads -- regression replay skipped)")
        return
    tally = defaultdict(lambda: [0, 0, 0])  # setup -> [danger caught, controls alerting, controls FULL]
    n_d = n_c = 0
    lost, silenced = [], []
    for c, clips in data.items():
        for name, clip in clips.items():
            danger = clip["danger"]
            n_d += danger
            n_c += not danger
            _, events = R.replay(clip)
            segs = [t for ts in clip["paths"].values() for t in ts]
            rules = "TRIGGER_ALL" if any(e.decision == "TRIGGER_ALL" for _, e in events) else ("TRIGGER_VERIFY" if events else "nothing")
            clf_hit = bool(len(segs)) and bool((model.score(segs) >= model.thr).any())
            kept = [e for _, e in events if e.layer.startswith("exact") or e.confidence >= WEAK_CONF
                    or model.score([e.transcript])[0] >= model.thr]
            gated = "TRIGGER_ALL" if any(e.decision == "TRIGGER_ALL" for e in kept) else ("TRIGGER_VERIFY" if kept else "nothing")
            for setup, tier in (("rules only (today)", rules), ("classifier only", "TRIGGER_VERIFY" if clf_hit else "nothing"),
                                ("rules + classifier gate on weak alerts", gated)):
                if danger:
                    tally[setup][0] += tier != "nothing"
                else:
                    tally[setup][1] += tier != "nothing"
                    tally[setup][2] += tier == "TRIGGER_ALL"
            if danger and rules != "nothing" and gated == "nothing":
                lost.append((c, name))
            if not danger and rules != "nothing" and gated == "nothing":
                silenced.append((c, name))
    print(f"\nREGRESSION (prepped_data x {len(data)} conditions -- NOT a generalisation measure)")
    print(f"  {'setup':<40} danger caught   controls alerting   controls FULL")
    for s, (d, ca, cf) in tally.items():
        print(f"  {s:<40} {d}/{n_d:<12}   {ca}/{n_c:<14}   {cf}")
    print(f"  gate: controls silenced {len(silenced)}: {sorted({n[:45] for _, n in silenced})}")
    print(f"  gate: !! danger LOST {len(lost)}: {lost}")


if __name__ == "__main__":
    m = train()
    regression(m)
