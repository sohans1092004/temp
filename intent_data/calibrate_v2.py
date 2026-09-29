"""Pick the v2 gate's thresholds on DEV (Kaggle GPU, ~2 min). Run from the bundle root:
    python -m intent_data.calibrate_v2 [model_dir]

Scores every DEV row of intent_dataset_v2.csv through intent_gate (premise = context + text,
exactly as live) and prints:
  1. veto threshold t (UDK_INTENT_THRESHOLD): an alert is silenced when P < t. Must keep
     >= 99.5% of dev distress (the v1 rule); more harmless vetoed is the gain.
  2. one-word support s (UDK_INTENT_SHORT_SUPPORT): a one-word hit alerts only when P >= s.
     Shown for one-word dev rows with and without context.
  3. per-category flag rates at the chosen t, including the new v2 categories.
TEST is not touched here -- it stays for the final report.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import intent_gate as IG  # noqa: E402

VETO = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
SUPPORT = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
KEEP = 0.995


def main() -> None:
    import torch

    model = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "finetuned_v2" / "mDeBERTa-v3-base-mnli-xnli")
    gate = IG.IntentGate(model, device="cuda" if torch.cuda.is_available() else "cpu")
    dev = [r for r in csv.DictReader(open(HERE / "intent_dataset_v2.csv", encoding="utf-8")) if r["split"] == "dev"]
    p = np.array([gate.p_distress(r["text"], r["context"]) for r in dev])
    y = np.array([int(r["label"]) for r in dev])
    print(f"dev rows: {len(dev)} ({int(y.sum())} distress)\n")

    print("1. veto threshold: dev distress KEPT (must be >= 99.5%) | dev harmless VETOED")
    best = None
    for t in VETO:
        kept, vetoed = (p[y == 1] >= t).mean(), (p[y == 0] < t).mean()
        ok = kept >= KEEP
        best = t if ok else best
        print(f"   t={t:<6} kept {kept:7.2%}  vetoed {vetoed:7.2%}  {'ok' if ok else 'LOSES distress'}")
    print(f"   -> highest t keeping >= 99.5%: {best}")
    lost = sorted(((p[i], dev[i]) for i in np.where((y == 1) & (p < (best or 0)))[0]), key=lambda x: x[0])
    for pi, r in lost[:10]:
        print(f"      lost p={pi:.3f} [{r['category']}] ctx={r['context'][:50]!r} | {r['text'][:70]}")

    one = np.array([IG.n_words(r["text"]) <= IG.SHORT_MAX_WORDS for r in dev])
    ctx = np.array([bool(r["context"]) for r in dev])
    print(f"\n2. one-word support ({one.sum()} one-word dev rows): danger KEPT | harmless SILENCED")
    for s in SUPPORT:
        cells = []
        for name, m in (("with context", one & ctx), ("no context", one & ~ctx)):
            d, h = m & (y == 1), m & (y == 0)
            cells.append(f"{name}: kept {(p[d] >= s).sum()}/{d.sum()}, silenced {(p[h] < s).sum()}/{h.sum()}")
        print(f"   s={s:<5} " + " | ".join(cells))

    t = best or 0.02
    print(f"\n3. per category at t={t}: % of rows the gate would VETO (harmless: high is good; distress: must be ~0)")
    per = defaultdict(lambda: [0, 0])
    for r, pi in zip(dev, p):
        k = f"{r['category'].split('+')[0]} [{r['label']}]"
        per[k][0] += int(pi < t)
        per[k][1] += 1
    for k in sorted(per):
        v, n = per[k]
        print(f"   {k:<34} {v:>5}/{n:<5} ({v / n:.0%})")


if __name__ == "__main__":
    main()
