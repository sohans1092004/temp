"""Zero-shot small instruct LLMs as a UDK-intent signal (text only, no training).
The model is asked whether the speaker is in danger / asking for help; score =
P(Yes) / (P(Yes) + P(No)) from the first answer token. Run from the Model folder (GPU):
    python -m intent_data.llm_eval [model ...]

Same protocol as nli_eval.py (dev threshold at >= 95% recall, scored once on TEST per
category, prepped replay as REGRESSION ONLY), so the results sit side by side.
Latency here is GPU, batch of 1, per window (these models are too slow for CPU).
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

from nli_eval import CONDS, LOGS, regression, report, thr_for  # noqa: E402

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"]  # Apache-2.0, MIT
OUT = HERE / "llm_results.json"
# the label policy from seeds.py, stated once; no examples from the dataset
SYSTEM = ("You check speech transcripts from a personal safety app. Answer Yes only if the speaker "
          "themselves is in danger or is asking for help right now. Answer No if the danger is denied "
          "or negated, quoted or reported from someone else, a joke or exaggeration, about an object "
          "or a task, or ordinary conversation. Transcripts may contain speech-recognition errors. "
          "Answer with one word: Yes or No.")


class LLMScorer:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda").eval()

        def ids(words):  # first token of each spelling ("Yes", " Yes", "yes"...), deduped
            return sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in words})

        yes, no = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
        # a SentencePiece tokenizer can split " Yes" into a bare "▁" + "Yes": that shared
        # first token would count for both answers, so it is dropped
        self.yes, self.no = [i for i in yes if i not in no], [i for i in no if i not in yes]
        print(f"  {name}: yes ids {self.yes}, no ids {self.no}", flush=True)

    def prompt(self, text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f'Transcript: "{text}"\nIs the speaker in danger or asking for help right now?'}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, texts: list[str], batch: int = 16) -> np.ndarray:
        out = []
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok([self.prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
                logp = torch.log_softmax(self.model(**enc).logits[:, -1].float(), dim=-1)
                y = torch.logsumexp(logp[:, self.yes], dim=-1)
                n = torch.logsumexp(logp[:, self.no], dim=-1)
                out.append(torch.sigmoid(y - n).cpu().numpy())  # P(Yes) among {Yes, No}
        return np.concatenate(out) if out else np.zeros(0)


def evaluate(name: str, rows: dict, replay) -> dict:
    m = LLMScorer(name)
    p = {s: m.score([r["text"] for r in v]) for s, v in rows.items()}
    y = {s: np.array([int(r["label"]) for r in v]) for s, v in rows.items()}
    thr = thr_for(p["dev"], y["dev"])
    hit = p["test"] >= thr
    per = defaultdict(lambda: [0, 0])
    for r, h in zip(rows["test"], hit):
        c = r["category"] + (f" [{r['label']}]" if r["category"].startswith("minimal_pair") else "")
        per[c][0] += int(h)
        per[c][1] += 1
    sample = [r["text"] for r in rows["test"][:40]]
    t0 = time.perf_counter()
    for s in sample:
        m.score([s], batch=1)
    ms = (time.perf_counter() - t0) / len(sample) * 1000
    res = {"threshold": thr, "test_recall": float(hit[y["test"] == 1].mean()), "test_fpr": float(hit[y["test"] == 0].mean()),
           "per_category": dict(per), "gpu_ms_per_window": round(ms, 1), "cpu_ms_per_window": None}
    if replay is not None:
        res["regression"] = regression(m, thr, replay)
    del m
    import torch
    torch.cuda.empty_cache()
    return res


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
              f"GPU {r['gpu_ms_per_window']} ms/window", flush=True)
    report(results)


if __name__ == "__main__":
    main()
