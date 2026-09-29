"""build_v2 merge rules, on temp files. python -m intent_data.test_build_v2"""
import csv
import json
import tempfile
from pathlib import Path

from .build_v2 import build, norm

d = Path(tempfile.mkdtemp())
(d / "llm").mkdir()
(d / "misheard").mkdir()
with open(d / "base.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["text", "label", "udk_id", "category", "template_id", "split", "policy_sensitive", "near_prepped"])
    w.writeheader()
    w.writerow({"text": "help me please", "label": "1", "udk_id": "UDK_08", "category": "exact", "template_id": "x",
                "split": "train", "policy_sensitive": "False", "near_prepped": "False"})
lines = [
    '```json',
    json.dumps({"context": "he is coming closer", "text": "Back!", "label": 1, "udk_id": "", "category": "short_with_context"}),
    json.dumps({"context": "your knight is hanging", "text": "Back!", "label": 0, "udk_id": "UDK_09", "category": "short_with_context"}),
    json.dumps({"context": "", "text": "Okay, let's go now.", "label": 0, "udk_id": "UDK_10", "category": "casual_udk_words"}),  # near a dev FA
    json.dumps({"context": "", "text": "Let go of me, you're hurting my arm.", "label": 1, "udk_id": "UDK_10", "category": "hard_cry"}),  # shares words only
    json.dumps({"context": "", "text": "help me please", "label": 1, "udk_id": "UDK_08", "category": "p4"}),  # dup of base
    json.dumps({"context": "x", "text": "", "label": 1}),  # no text
    "not json at all",
    '```',
]
(d / "llm" / "p2.jsonl").write_text("\n".join(lines), encoding="utf-8")
(d / "misheard" / "fleurs_hi.csv").write_text("text,source,udk_hit\nThanks for watching!,fleurs_hi,\n", encoding="utf-8")

rows = build(d / "base.csv", d / "llm", d / "misheard", ["Okay, let's go.", "back.", "Let me, let me, let me."])
new = [r for r in rows if r["template_id"] != "x"]
clean = [r for r in new if "+asr_noise" not in r["category"]]
assert {(r["context"], r["text"], r["label"]) for r in clean} == {
    ("he is coming closer", "Back!", "1"), ("your knight is hanging", "Back!", "0"), ("", "Thanks for watching!", "0"),
    ("", "Let go of me, you're hurting my arm.", "1")}, clean
pos = next(r for r in clean if r["label"] == "1" and r["context"])
neg = next(r for r in clean if r["label"] == "0" and r["context"])
assert pos["udk_id"] == "INTENT" and neg["udk_id"] == "" and neg["resembles"] == "UDK_09"
assert pos["split"] == neg["split"]  # both halves of a context pair in one split
assert all(r["split"] == next(c["split"] for c in clean if norm(c["text"]) == norm(r["text"])) for r in new if norm(r["text"]) == "back")
assert next(r for r in clean if r["category"] == "misheard_fleurs_hi")["label"] == "0"
print("test_build_v2: all checks passed")
