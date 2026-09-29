"""Replay logged transcripts (Colab evaluate_prepped_data logs) through udk_engine
matching rules on CPU -- no audio models, only the MiniLM text matcher.
Step 1: with unchanged rules, must reproduce each clip's logged tier.
Step 2: apply candidate false-alarm fixes; any lost danger detection = rejected."""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())  # Model folder
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import udk_engine
from udks import GENERAL_UDKS

LOGS = r"C:\Users\sohan\Downloads"
CONDS = ["normal", "quiet", "loud", "crowd", "pocket", "phone"]
RANK = {"nothing": 0, "TRIGGER_VERIFY": 1, "TRIGGER_ALL": 2}


def parse(path):
    """-> {clip: {"danger": bool, "logged": tier, "paths": {stt: [segment transcripts in order]}}}"""
    text = open(path, encoding="utf-8", errors="replace").read()
    clips = {}
    for block in text.split("=" * 100):
        lines = [l for l in block.strip().splitlines()]
        # decoder warnings (librosa/mpg123) interleave with stdout on Colab and some
        # clips lose their "expected:" or "RESULT:" line. Clip name = line 0; a block is
        # a clip if ANY of its marker lines survived (the danger label comes from the summary).
        if not lines or not any(l.strip().startswith(("expected:", "RESULT:", "heard")) for l in lines):
            continue
        name, paths = lines[0].strip(), defaultdict(list)
        for i, l in enumerate(lines):
            m = re.match(r"\s+heard(?: \[(\w+)\])?: (.*)$", l)
            if not m or i + 1 >= len(lines):
                continue
            decision = lines[i + 1].strip()
            if "/sentence" in decision:  # a per-sentence extra row, not a segment transcript
                continue
            paths[m.group(1) or "whisper"].append(eval(m.group(2)))  # repr() of the transcript
        clips[name] = {"paths": dict(paths)}
    on = False
    for line in text.splitlines():
        on = on or line.startswith("ALERT-LEVEL RESULT")
        m = on and re.match(r"^\s+(DANGER|control)\s+(\S+)\s+conf (.{14}) (.+)$", line)
        if m:
            # the summary prints stem[:70]; several stems share long prefixes, so match exactly that
            key = next((k for k in clips if k[:70].strip() == m.group(4).strip()), None)
            if key:
                clips[key].update(danger=m.group(1) == "DANGER", logged=m.group(2))
    return clips


_semantic = None


def semantic():
    global _semantic
    if _semantic is None:
        from semantic import SentenceTransformerSemanticMatcher

        _semantic = SentenceTransformerSemanticMatcher(device="cpu")
    return _semantic


def replay(clip, post=lambda e: e):
    """Clip tier + events: every STT path decided with its own engine, like run_file_either."""
    events = []
    for path, segs in clip["paths"].items():
        engine = udk_engine.UDKEngine(GENERAL_UDKS, semantic_matcher=semantic())
        for i, text in enumerate(segs):
            for e in engine.decide_all(text, now_s=1000.0 + i * 0.1):
                e = post(e)
                if e.decision != "NO_ACTION":
                    events.append((path, e))
    tier = "TRIGGER_ALL" if any(e.decision == "TRIGGER_ALL" for _, e in events) else ("TRIGGER_VERIFY" if events else "nothing")
    return tier, events


def load_all():
    return {c: parse(os.path.join(LOGS, f"prepped_gpu_{c}.log")) for c in CONDS}


if __name__ == "__main__":
    data = load_all()
    agree = total = 0
    for c, clips in data.items():
        for name, clip in clips.items():
            tier, _ = replay(clip)
            total += 1
            agree += tier == clip.get("logged")
            if tier != clip.get("logged"):
                print(f"  MISMATCH {c:<7} {name[:55]:<55} logged={clip.get('logged')} replay={tier}")
    print(f"\nbaseline replay reproduces {agree}/{total} logged clip tiers")
