"""Fine-tune NLI cross-encoders as a distress-intent classifier on the generated dataset
(Colab GPU). Run from the Model folder:
    python -m intent_data.finetune [model ...]

Each model keeps its NLI input format -- (transcript window, hypothesis) -- so training
starts from its entailment knowledge; the 3-way NLI head is replaced by a 2-way head.
Protocol (same as nli_eval / train_classifier, fixed in advance):
  - train on TRAIN, pick the epoch on DEV loss, threshold on DEV (recall >= 95%)
  - scored once on TEST per category (minimal pairs split by label, heavy ASR errors...)
  - REGRESSION ONLY: prepped transcripts replayed (alone, and as a gate on weak alerts)
  - REAL false alarms: if real_world/real_world_alerts.csv exists, % of those real
    transcripts the model would still flag (all are non-distress)
"""
from __future__ import annotations

import csv
import os
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

from nli_eval import CONDS, HYPOTHESIS, LOGS, regression, thr_for  # noqa: E402
from intent_gate import premise  # noqa: E402  (the live gate's input format -- train on exactly that)

MODELS = ["cross-encoder/nli-deberta-v3-small", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"]
EPOCHS, LR, BATCH, MAX_LEN, SEED = 3, 2e-5, 16, 160, 0
# INTENT_DATASET=intent_data/intent_dataset_v2.csv (build_v2.py: context column) -> results/weights go to *_v2
DATASET = Path(os.environ.get("INTENT_DATASET") or HERE / "intent_dataset.csv")
V2 = DATASET.name != "intent_dataset.csv"
OUT = HERE / ("finetune_results_v2.json" if V2 else "finetune_results.json")
SAVE = HERE / ("finetuned_v2" if V2 else "finetuned")
PAIR_CATS = ("minimal_pair", "short_with_context", "context_pair")  # reported per label


def inp(r: dict) -> str:
    return premise(r["text"], r.get("context") or "")
REAL_ALERTS = [Path(p) for p in [os.environ.get("INTENT_REAL_ALERTS", "")] if p] + [
    HERE.parent / "real_world" / "real_world_alerts.csv", Path("/content/drive/MyDrive/real_world/real_world_alerts.csv")]


class FineTuned:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name = torch, name
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=2, ignore_mismatched_sizes=True).float().to(self.dev)  # fp32 master weights; some checkpoints ship fp16

    def _enc(self, texts):
        return self.tok(list(texts), [HYPOTHESIS] * len(texts), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(self.dev)

    def score(self, texts: list[str], batch: int = 64) -> np.ndarray:
        self.model.eval()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                logits = self.model(**self._enc(texts[i:i + batch])).logits
                out.append(self.torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, train, dev) -> dict:
        torch = self.torch
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        y = np.array([int(r["label"]) for r in train])
        w = torch.tensor([1.0, float((1 - y.mean()) / y.mean())], device=self.dev)  # balance classes (float32, like logits)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * int(np.ceil(len(train) / BATCH))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
        scaler = torch.amp.GradScaler(enabled=self.dev == "cuda")
        yd = torch.tensor([int(r["label"]) for r in dev], device=self.dev)
        best, best_state, log = float("inf"), None, []
        for epoch in range(EPOCHS):
            self.model.train()
            order = rng.permutation(len(train))
            for i in range(0, len(order), BATCH):
                b = [train[j] for j in order[i:i + BATCH]]
                with torch.autocast(self.dev, enabled=self.dev == "cuda"):
                    logits = self.model(**self._enc([inp(r) for r in b])).logits
                    loss = loss_fn(logits.float(), torch.tensor([int(r["label"]) for r in b], device=self.dev))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            self.model.eval()
            with torch.no_grad():
                dl = []
                for i in range(0, len(dev), 64):
                    dl.append(loss_fn(self.model(**self._enc([inp(r) for r in dev[i:i + 64]])).logits.float(),
                                      yd[i:i + 64]).item() * len(dev[i:i + 64]))
                dev_loss = sum(dl) / len(dev)
            log.append(round(dev_loss, 4))
            print(f"    epoch {epoch + 1}/{EPOCHS}: dev loss {dev_loss:.4f}", flush=True)
            if dev_loss < best:
                best, best_state = dev_loss, {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return {"dev_loss_per_epoch": log, "best_dev_loss": round(best, 4)}


def real_false_alarms(m, thr) -> dict | None:
    path = next((p for p in REAL_ALERTS if p.exists()), None)
    if not path:
        return None
    texts = [r["heard"] for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("heard", "").strip()]
    if not texts:
        return None
    flagged = m.score(texts) >= thr
    return {"n": len(texts), "still_flagged": int(flagged.sum()),
            "examples_cleared": [t for t, f in zip(texts, flagged) if not f][:8],
            "examples_still_flagged": [t for t, f in zip(texts, flagged) if f][:8]}


def main() -> None:
    names = sys.argv[1:] or MODELS
    global EPOCHS, OUT
    rows = defaultdict(list)
    print(f"dataset: {DATASET}", flush=True)
    for r in csv.DictReader(open(DATASET, encoding="utf-8")):
        rows[r["split"]].append(r)
    if os.environ.get("INTENT_SMOKE") == "1":  # ~1 min: proves the loop runs; numbers mean nothing
        rng = np.random.default_rng(0)
        rows = {s: [v[i] for i in rng.choice(len(v), 64, replace=False)] for s, v in rows.items()}
        EPOCHS, OUT = 1, HERE / "finetune_smoke.json"
    replay = None
    if all((LOGS / f"prepped_gpu_{c}.log").exists() for c in CONDS):
        import scratch_replay as R

        replay = {c: R.parse(str(LOGS / f"prepped_gpu_{c}.log")) for c in CONDS}
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        if name in results:
            print(f"[skip] {name} (done)", flush=True)
            continue
        print(f"\n== fine-tuning {name}", flush=True)
        t0 = time.time()
        m = FineTuned(name)
        info = m.fit(rows["train"], rows["dev"])
        pd = m.score([inp(r) for r in rows["dev"]])
        thr = thr_for(pd, np.array([int(r["label"]) for r in rows["dev"]]))
        pt = m.score([inp(r) for r in rows["test"]])
        yt = np.array([int(r["label"]) for r in rows["test"]])
        hit = pt >= thr
        per = defaultdict(lambda: [0, 0])
        for r, h in zip(rows["test"], hit):
            c = r["category"] + (f" [{r['label']}]" if r["category"].startswith(PAIR_CATS) else "")
            per[c][0] += int(h)
            per[c][1] += 1
        t1 = time.perf_counter()
        for r in rows["test"][:50]:
            m.score([inp(r)], batch=1)
        res = {**info, "threshold": thr, "test_recall": float(hit[yt == 1].mean()), "test_fpr": float(hit[yt == 0].mean()),
               "per_category": dict(per), "ms_per_window": round((time.perf_counter() - t1) / 50 * 1000, 1),
               "device": m.dev, "train_minutes": round((time.time() - t0) / 60, 1)}
        if replay:
            res["regression"] = regression(m, thr, replay)
        res["real_false_alarms"] = real_false_alarms(m, thr)
        results[name] = res
        OUT.write_text(json.dumps(results, indent=1))
        SAVE.mkdir(exist_ok=True)
        m.model.save_pretrained(SAVE / name.split("/")[-1])
        m.tok.save_pretrained(SAVE / name.split("/")[-1])
        print(f"  thr {thr:.2f} | TEST recall {res['test_recall']:.1%}, FPR {res['test_fpr']:.1%} | {res['ms_per_window']} ms/window on {m.dev}", flush=True)
        del m
        import gc

        gc.collect()
    from nli_eval import report

    report(results)
    for n, r in results.items():
        rf = r.get("real_false_alarms")
        if rf:
            print(f"\nREAL YouTube false-alarm transcripts, {n}: still flagged {rf['still_flagged']}/{rf['n']}")
            print("   cleared e.g.:", rf["examples_cleared"][:4])
            print("   still flagged e.g.:", rf["examples_still_flagged"][:4])


if __name__ == "__main__":
    main()
