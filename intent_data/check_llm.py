"""Catch the two ways LLM pair data goes wrong, before it reaches training:
    python -m intent_data.check_llm intent_data/llm/p5.jsonl
- REPEATS: the same context (or same 4-word opening) used again and again
- REASON-ADDED pairs: the label-0 context is the label-1 context plus a reason
  ("Come closer." / "Come closer, the baby is sleeping.") -- teaches "bare command = danger".
"""
import sys
from collections import Counter

from .build_v2 import norm, parse_jsonl


def check(rows: list[dict]) -> dict:
    ctx = [norm(r["context"]) for r in rows]
    openings = Counter(" ".join(c.split()[:4]) for c in ctx if c)
    by_text = {}
    for r, c in zip(rows, ctx):
        by_text.setdefault(norm(r["text"]), {}).setdefault(r["label"], []).append(c)
    reason_added = sum(1 for d in by_text.values() for c1 in d.get(1, []) for c0 in d.get(0, [])
                       if c1 and c0 != c1 and c0.startswith(" ".join(c1.split()[:4])))
    # "Amma, leave me alone, I'm studying" -> "Leave me alone!": the cry's own words in the
    # context teach "words already said = harmless" (and the speaker is usually the crier)
    cry_in_context = [r["text"] for r, c in zip(rows, ctx) if norm(r["text"]) and norm(r["text"]) in c]
    return {"rows": len(rows), "duplicate contexts": len(ctx) - len(set(ctx)),
            "openings used >2x": {k: v for k, v in openings.items() if v > 2},
            "reason-added pairs": reason_added, "cry inside context": cry_in_context}


if __name__ == "__main__":
    from pathlib import Path

    for p in sys.argv[1:]:
        res = check(parse_jsonl(Path(p)))
        bad = res["duplicate contexts"] or res["openings used >2x"] or res["reason-added pairs"] or res["cry inside context"]
        print(f"{p}: {'FIX' if bad else 'OK'}  {res}")
