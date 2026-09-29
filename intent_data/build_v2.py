"""intent_dataset_v2.csv = intent_dataset.csv (context "") + the LLM prompt outputs
(intent_data/llm/*.jsonl) + STT-invented English (intent_data/misheard/*.csv, from
harvest_misheard.py). Run from the Model folder:  python -m intent_data.build_v2

- Split by the CURRENT sentence (normalised), so both halves of a context pair and the
  ASR-noise copy always land in the same split -- dev/test sentences are never seen in training.
- Anything close to a DEV false alarm (dev_false_alarms.txt: the real alerts we already
  studied) is dropped: those stay a measurement, never training data. Only sentences of
  3+ words are compared -- "back." / "go." are generic words, not copied sentences.
- label 1 rows get udk_id "INTENT" when the LLM gave none (test_dataset's invariant:
  positive <=> has a UDK id); a label-0 row keeps the UDK it resembles in `resembles`.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from .generate import asr_noise, bucket

HERE = Path(__file__).parent
OUT = HERE / "intent_dataset_v2.csv"
COLS = ["context", "text", "label", "udk_id", "resembles", "category", "template_id", "split", "policy_sensitive", "near_prepped"]
NEAR_DEV_FA = 85  # fuzz.ratio on the whole normalised sentence: a near-COPY, not shared words


def norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def parse_jsonl(path: Path) -> list[dict]:
    """Lenient: LLM output may carry code fences, blank lines or a stray comment line."""
    out = []
    for n, line in enumerate(open(path, encoding="utf-8")):
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [skip] {path.name}:{n + 1} not JSON", file=sys.stderr)
            continue
        text, label = str(r.get("text", "")).strip(), r.get("label")
        if not text or str(label) not in ("0", "1"):
            print(f"  [skip] {path.name}:{n + 1} missing text/label", file=sys.stderr)
            continue
        out.append({"context": str(r.get("context") or "").strip(), "text": text, "label": int(label),
                    "udk": str(r.get("udk_id") or "").strip(), "category": str(r.get("category") or path.stem).strip()})
    return out


def build(base_csv: Path, llm_dir: Path, misheard_dir: Path, dev_fa: list[str], seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = [{**r, "context": "", "resembles": ""} for r in csv.DictReader(open(base_csv, encoding="utf-8"))]
    new = []
    for p in sorted(llm_dir.glob("*.jsonl")) if llm_dir.exists() else []:
        for r in parse_jsonl(p):
            pos = r["label"] == 1
            new.append({"context": r["context"], "text": r["text"], "label": str(r["label"]),
                        "udk_id": (r["udk"] or "INTENT") if pos else "", "resembles": "" if pos else r["udk"],
                        "category": r["category"], "template_id": f"llm/{p.stem}"})
    for p in sorted(misheard_dir.glob("*.csv")) if misheard_dir.exists() else []:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("text", "").strip():
                new.append({"context": "", "text": r["text"].strip(), "label": "0", "udk_id": "",
                            "resembles": r.get("udk_hit", ""), "category": f"misheard_{r.get('source', p.stem)}",
                            "template_id": f"misheard/{p.stem}"})
    dev_long = [norm(t) for t in dev_fa if len(norm(t).split()) >= 3]
    base_split = {norm(r["text"]): r["split"] for r in rows}
    seen = {(norm(r["context"]), norm(r["text"])) for r in rows}
    kept, dropped = [], Counter()
    for r in new:
        key = (norm(r["context"]), norm(r["text"]))
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if len(key[1].split()) >= 3 and any(fuzz.ratio(key[1], d) >= NEAR_DEV_FA for d in dev_long):
            dropped["near a dev false alarm"] += 1
            continue
        seen.add(key)
        split = base_split.get(key[1]) or bucket(key[1])
        r.update(split=split, policy_sensitive="False", near_prepped="False")
        kept.append(r)
        noisy = {**r, "context": asr_noise(r["context"], rng) if r["context"] else "", "text": asr_noise(r["text"], rng),
                 "category": r["category"] + "+asr_noise"}
        if (norm(noisy["context"]), norm(noisy["text"])) not in seen and noisy["text"] != r["text"]:
            seen.add((norm(noisy["context"]), norm(noisy["text"])))
            kept.append(noisy)
    if dropped:
        print(f"  dropped: {dict(dropped)}")
    return rows + kept


def main() -> None:
    dev_fa_file = HERE / "dev_false_alarms.txt"
    dev_fa = [l.strip() for l in open(dev_fa_file, encoding="utf-8") if l.strip() and not l.startswith("#")]
    rows = build(HERE / "intent_dataset.csv", HERE / "llm", HERE / "misheard", dev_fa)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    c = Counter((r["category"].split("+")[0], r["label"], r["split"]) for r in rows)
    cats = sorted({k[0] for k in c})
    print(f"{OUT.name}: {len(rows)} rows ({sum(r['context'] != '' for r in rows)} with context)")
    print(f"{'category':<26}{'label':>6}{'train':>8}{'dev':>6}{'test':>6}")
    for cat in cats:
        for lab in ("0", "1"):
            if any(c[(cat, lab, s)] for s in ("train", "dev", "test")):
                print(f"{cat:<26}{lab:>6}" + "".join(f"{c[(cat, lab, s)]:>{8 if s == 'train' else 6}}" for s in ("train", "dev", "test")))


if __name__ == "__main__":
    main()
