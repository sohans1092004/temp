"""Invariants of intent_dataset.csv. python -m intent_data.test_dataset (after generate)."""
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).parent / "intent_dataset.csv", encoding="utf-8")))
splits_of = defaultdict(set)
for r in rows:
    for tid in r["template_id"].split("|"):
        splits_of[tid].add(r["split"])
    assert r["label"] in ("0", "1"), r
    assert (r["label"] == "1") == bool(r["udk_id"]), f"positive <=> has a UDK id: {r}"
    assert r["policy_sensitive"] == "False" or r["category"].split("+")[0] in ("quote", "own_speech_nonthreat", "retraction"), r
    assert not (r["near_prepped"] == "True" and r["split"] != "train"), f"near-prepped text leaked into {r['split']}: {r}"

# no template in two splits -> dev/test patterns are genuinely unseen
leaks = {t: s for t, s in splits_of.items() if len(s) > 1}
assert not leaks, f"templates spanning splits: {list(leaks.items())[:5]}"
# no sentence in two splits
texts = defaultdict(set)
for r in rows:
    texts[r["text"].lower()].add(r["split"])
assert all(len(s) == 1 for s in texts.values())
# every UDK: exact phrase in train, and unseen paraphrases in dev AND test
for u in {r["udk_id"] for r in rows if r["udk_id"] and r["udk_id"] != "INTENT"}:  # INTENT = minimal-pair distress, no UDK
    cats = {(r["category"].split("+")[0], r["split"]) for r in rows if r["udk_id"] == u}
    assert ("exact", "train") in cats and ("paraphrase", "dev") in cats and ("paraphrase", "test") in cats, (u, cats)
# every hard-negative category is represented in dev and test
for c in ("negation", "quote", "hyperbole", "instruction", "near_miss", "chatter", "minimal_pair_neg", "negation_danger"):
    for s in ("dev", "test"):
        assert any(r["category"].split("+")[0] == c and r["split"] == s for r in rows), (c, s)
# minimal pairs: both sides of every phrase in ONE split, and both sides present in dev and test
pair_splits = defaultdict(set)
for r in rows:
    if r["category"].startswith("minimal_pair"):
        pair_splits[r["template_id"]].add((r["split"], r["label"]))
for tid, s_l in pair_splits.items():
    assert len({s for s, _ in s_l}) == 1 and {l for _, l in s_l} == {"0", "1"}, (tid, s_l)
for s in ("dev", "test"):
    labels = {r["label"] for r in rows if r["category"].startswith("minimal_pair") and r["split"] == s}
    assert labels == {"0", "1"}, (s, labels)
print(f"intent_data: all checks passed ({len(rows)} rows)")
