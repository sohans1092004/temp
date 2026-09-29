"""Pick the intent veto threshold (UDK_INTENT_THRESHOLD) on the intent dataset's DEV split.
A veto must only fire when the model is confident a sentence is NOT distress, so the
threshold is the highest one that still keeps >= 99.5% of dev distress sentences
(0.99 was finetune.py's DETECTION threshold at 95% recall: as a veto it silenced real
cries, 2026-09-26 Kaggle run). Nothing from prepped_data is used. GPU run:
    UDK_INTENT_GATE=<mDeBERTa dir> python calibrate_veto.py
"""
import csv
import os
from pathlib import Path

import intent_gate

VETO_RECALL = 0.995
GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

rows = [r for r in csv.DictReader(open(Path(__file__).parent / "intent_data" / "intent_dataset.csv", encoding="utf-8"))
        if r["split"] in ("dev", "test")]
dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
gate = intent_gate.IntentGate(os.environ["UDK_INTENT_GATE"], device=dev)
for r in rows:
    r["p"] = gate.p_distress(r["text"])

print(f"{'threshold':>9} | {'dev: distress kept':>18} {'harmless vetoed':>16} | {'test: distress kept':>19} {'harmless vetoed':>16}")
best = None
for t in GRID:
    cells = []
    for split in ("dev", "test"):
        pos = [r for r in rows if r["split"] == split and r["label"] == "1"]
        neg = [r for r in rows if r["split"] == split and r["label"] == "0"]
        kept = sum(r["p"] >= t for r in pos) / len(pos)
        cells += [kept, sum(r["p"] < t for r in neg) / len(neg)]
    if cells[0] >= VETO_RECALL:
        best = t
    print(f"{t:>9} | {cells[0]:>18.1%} {cells[1]:>16.1%} | {cells[2]:>19.1%} {cells[3]:>16.1%}")
print(f"\n=> UDK_INTENT_THRESHOLD={best}  (highest with dev distress kept >= {VETO_RECALL:.1%})")
print("\nlowest-scoring DEV distress sentences (what a higher threshold would silence):")
for r in sorted((r for r in rows if r["split"] == "dev" and r["label"] == "1"), key=lambda r: r["p"])[:12]:
    print(f"   p={r['p']:.4f}  {r['category']:<28} {r['text'][:80]!r}")
