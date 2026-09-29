"""Template-generated text dataset for a UDK-intent classifier (free, deterministic).
Run from the Model folder:  python -m intent_data.generate
-> intent_data/intent_dataset.csv  (+ stats printed)

Columns: text, label (1 = distress request by the speaker), udk_id, category,
template_id, split (train/dev/test), policy_sensitive, near_prepped

Splits are by TEMPLATE, never by sentence: dev/test contain sentence patterns (and
UDK paraphrases) the model never saw, so their scores measure generalisation.
Anything too close to a prepped_data transcript is kept out of dev/test.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import random
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Model folder (udks.py)
from udks import GENERAL_UDKS  # noqa: E402

from . import seeds as S  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "intent_dataset.csv"
SEED = 0
PER_TEMPLATE = 40  # slot fillings per negative template (negatives should outnumber positives ~3:1)
# label-neutral wrappers for negation pairs (never "don't worry"/"help": no label hints)
NEG_PAIR_WRAP = ["{x}", "listen, {x}", "mom, {x}", "{x}, I'm telling you", "{x}, okay?", "hello? {x}", "{x}, can you hear me"]
PREPPED_LOGS =[Path.home() / "Downloads" / f"prepped_gpu_{c}.log" for c in ("normal", "quiet", "loud", "crowd", "pocket", "phone")]


def bucket(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("dev" if h < 85 else "test")


def split_templates(ids: list[str]) -> dict[str, str]:
    """70/15/15 of a category's TEMPLATES by stable hash rank, with >= 1 template in
    dev and in test whenever there are >= 3 (independent hashing can leave a split empty)."""
    order = sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())
    n = len(order)
    n_test = max(1, round(0.15 * n)) if n >= 3 else 0
    n_dev = max(1, round(0.15 * n)) if n >= 3 else 0
    return {i: ("test" if k < n_test else "dev" if k < n_test + n_dev else "train") for k, i in enumerate(order)}


def fill(template: str, rng: random.Random, k: int) -> list[str]:
    slots = re.findall(r"{(\w+)}", template)
    if not slots:
        return [template]
    combos = list(itertools.product(*[S.SLOTS[s] for s in slots]))
    rng.shuffle(combos)
    out = []
    for combo in combos[:k]:
        t = template
        for s, v in zip(slots, combo):
            t = t.replace("{" + s + "}", v, 1)
        out.append(t)
    return out


def asr_noise(text: str, rng: random.Random) -> str:
    """Transcript-like variation that never changes meaning: case, punctuation,
    fillers, contraction spelling, a repeated word. Never drops words (dropping
    'not' would silently flip a label)."""
    t = text.lower() if rng.random() < 0.7 else text
    if rng.random() < 0.6:
        t = t.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    if rng.random() < 0.3:
        t = rng.choice(["uh ", "um ", "so ", "like "]) + t
    if rng.random() < 0.3:
        t = t.replace("I'm", "I am").replace("don't", "dont").replace("i'm", "im")
    words = t.split()
    if len(words) > 2 and rng.random() < 0.25:
        i = rng.randrange(len(words))
        words.insert(i, words[i])
    return " ".join(words)


def rows() -> list[dict]:
    rng = random.Random(SEED)
    out = []

    def add(text, label, udk, cat, tid, split, policy=False):
        out.append({"text": text.strip(), "label": label, "udk_id": udk, "category": cat, "template_id": tid,
                    "split": split, "policy_sensitive": policy})

    exact = {u.udk_id: u.phrase for u in GENERAL_UDKS}
    # ---- positives: exact phrase always in train; paraphrases split per UDK so every
    #      UDK has unseen paraphrases in dev AND test
    for udk, paras in S.PARAPHRASES.items():
        paras = [p for p in dict.fromkeys(paras) if p.lower() != exact[udk].lower()]
        paras.sort(key=lambda p: hashlib.md5(p.encode()).hexdigest())
        splits = {paras[-1]: "test", paras[-2]: "dev"} if len(paras) >= 4 else {}
        for j, p in enumerate([exact[udk]] + paras):
            split = "train" if j == 0 else splits.get(p, "train")
            cat = "exact" if j == 0 else "paraphrase"
            tid = f"pos/{udk}/{j}"
            for pre, suf in rng.sample(list(itertools.product(S.POS_PREFIX, S.POS_SUFFIX)), 6):
                s = f"{pre}{p[0].lower() + p[1:] if pre else p}{suf}"
                add(s, 1, udk, cat, tid, split)
                add(asr_noise(s, rng), 1, udk, cat + "+asr_noise", tid, split)

    # ---- negatives
    def neg(cat, templates, k=PER_TEMPLATE, wrap=None, policy=False):
        splits = split_templates([f"{cat}/{i}" for i in range(len(templates))])
        for i, t in enumerate(templates):
            tid = f"{cat}/{i}"
            split = splits[tid]
            for s in fill(t, rng, k):
                for w in (wrap or ["{x}"]):
                    x = w.replace("{x}", s)
                    add(x, 0, "", cat, tid, split, policy)
                    if rng.random() < 0.5:
                        add(asr_noise(x, rng), 0, "", cat + "+asr_noise", tid, split, policy)

    neg("negation", S.NEGATION, wrap=S.NEG_WRAP)
    quotes = [f"{subj} {verb} '{exact[u]}' {end}" for subj in S.QUOTE_SUBJ for verb in S.QUOTE_VERB[:3]
              for u in exact for end in S.QUOTE_END[:2]]
    rng.shuffle(quotes)
    # quote templates grouped by quoted UDK so a UDK's quotes share a split
    by_udk = {}
    for q in quotes:
        u = next(k for k, v in exact.items() if f"'{v}'" in q)
        by_udk.setdefault(u, []).append(q)
    qsplit = split_templates([f"quote/{u}" for u in by_udk])
    for u, qs in by_udk.items():
        tid = f"quote/{u}"
        for q in qs[:12]:
            add(q, 0, "", "quote", tid, qsplit[tid], policy=True)
    neg("hyperbole", S.HYPERBOLE)
    neg("instruction", S.INSTRUCTION)
    neg("near_miss", S.NEAR_MISS)
    neg("own_speech_nonthreat", S.OWN_SPEECH_NONTHREAT, k=10, policy=True)
    neg("chatter", S.CHATTER, k=80)
    neg("hinglish", S.HINGLISH, k=40)

    # ---- negation minimal pairs (see seeds.NEG_PAIRS): split by UDK, both sides share
    # the split and the same wrappers, so only the negation separates the labels
    nsplit = split_templates([f"negpair/{u}" for u in S.NEG_PAIRS])
    for u, pairs in S.NEG_PAIRS.items():
        tid = f"negpair/{u}"
        for danger, negated in pairs:
            for w in rng.sample(NEG_PAIR_WRAP, 4):
                for label, s in ((1, danger), (0, negated)):
                    x = w.replace("{x}", s)
                    add(x, label, u if label else "", "minimal_pair_neg", tid, nsplit[tid])
                    add(asr_noise(x, rng), label, u if label else "", "minimal_pair_neg+asr_noise", tid, nsplit[tid])
    dsplit = split_templates([f"negdanger/{i}" for i in range(len(S.NEG_DANGER))])
    for i, s in enumerate(S.NEG_DANGER):
        tid = f"negdanger/{i}"
        for w in rng.sample(NEG_PAIR_WRAP, 3):
            x = w.replace("{x}", s)
            add(x, 1, "INTENT", "negation_danger", tid, dsplit[tid])
            add(asr_noise(x, rng), 1, "INTENT", "negation_danger+asr_noise", tid, dsplit[tid])


    # ---- composition: real speech segments often hold 2-3 sentences. Components are
    # only ever combined WITHIN one split, so a test pair contains test templates only.
    base = list(out)
    per_split = {"train": 1500, "dev": 300, "test": 300}
    for split, n_pairs in per_split.items():
        chat = [r for r in base if r["split"] == split and r["category"] in ("chatter", "hinglish")]
        hard = [r for r in base if r["split"] == split and r["label"] == 0 and r["category"].split("+")[0] not in ("chatter", "hinglish")]
        pos = [r for r in base if r["split"] == split and r["label"] == 1]
        if not chat:
            continue
        for _ in range(n_pairs):
            a, b = rng.sample(chat, 2)
            add(f"{a['text']}, {b['text']}", 0, "", "chatter+context", f"{a['template_id']}|{b['template_id']}", split)
        for r in hard + rng.sample(pos, len(pos) // 2):
            c = rng.choice(chat)
            first, second = (c, r) if rng.random() < 0.7 else (r, c)
            add(f"{first['text']}, {second['text']}", r["label"], r["udk_id"], r["category"].split("+")[0] + "+context",
                r["template_id"], split, r["policy_sensitive"])

    # ---- minimal pairs: same phrase, label decided by neighbouring words. A phrase's
    # contexts all share one split. "+delayed" puts an everyday sentence (same split)
    # between the phrase and its deciding context -- the 2-10 s later case.
    psplit = split_templates([f"pair/{p}" for p in S.MINIMAL_PAIRS])
    for phrase, (danger, harmless) in S.MINIMAL_PAIRS.items():
        tid = f"pair/{phrase}"
        split = psplit[tid]
        chat = [r["text"] for r in base if r["split"] == split and r["category"] == "chatter"]
        for label, contexts in ((1, danger), (0, harmless)):
            for ctx in contexts:
                udk = "INTENT" if label else ""
                add(f"{phrase}, {ctx}", label, udk, "minimal_pair", tid, split)
                if chat:
                    add(f"{phrase}. {rng.choice(chat)}. {ctx}", label, udk, "minimal_pair+delayed", tid, split)
    rsplit = split_templates([f"retraction/{i}" for i in range(len(S.RETRACTIONS))])
    for i, t in enumerate(S.RETRACTIONS):
        add(t, 0, "", "retraction", f"retraction/{i}", rsplit[f"retraction/{i}"], policy=True)

    # ---- heavy transcription errors (sound-alike word swaps, never negation words),
    # on a sample of everything; same split and label as the source sentence
    for r in list(out):
        if rng.random() < 0.3:
            noisy = asr_heavy(r["text"], rng)
            if noisy != r["text"]:
                add(noisy, r["label"], r["udk_id"], r["category"].split("+")[0] + "+asr_heavy", r["template_id"],
                    r["split"], r["policy_sensitive"])
    return out


def asr_heavy(text: str, rng: random.Random) -> str:
    """Swap 1-2 confusable words for sound-alikes Whisper/Parakeet produce on noisy speech."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if w.lower().strip(string.punctuation) in S.ASR_CONFUSIONS]
    for i in rng.sample(idx, min(len(idx), rng.choice([1, 2]))):
        key = words[i].lower().strip(string.punctuation)
        words[i] = rng.choice(S.ASR_CONFUSIONS[key])
    return " ".join(words).lower()


def prepped_transcripts() -> list[str]:
    texts = []
    for log in PREPPED_LOGS:
        if log.exists():
            for line in open(log, encoding="utf-8", errors="replace"):
                m = re.match(r"\s+heard(?: \[\w+\])?: (.*)$", line)
                if m:
                    try:
                        texts.append(str(eval(m.group(1))))
                    except Exception:
                        pass
    return list(dict.fromkeys(texts))


def main() -> None:
    from rapidfuzz import fuzz

    data = rows()
    # dedupe by normalised text; a sentence can only live in one split
    seen, uniq = set(), []
    for r in data:
        key = re.sub(r"[^a-z0-9 ]", "", r["text"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    prepped = prepped_transcripts()
    for r in uniq:
        r["near_prepped"] = any(fuzz.token_set_ratio(r["text"].lower(), p.lower()) >= 90 for p in prepped) if prepped else False
    # near-prepped sentences are DROPPED from dev/test (moving them into train would leak
    # a held-out template into training); in train they stay, flagged
    moved = sum(r["near_prepped"] and r["split"] != "train" for r in uniq)
    uniq = [r for r in uniq if not (r["near_prepped"] and r["split"] != "train")]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    print(f"{OUT.name}: {len(uniq)} sentences ({sum(r['label'] for r in uniq)} positive)")
    print(f"prepped transcripts checked: {len(prepped)}; near-duplicates: {sum(r['near_prepped'] for r in uniq)} "
          f"({moved} dropped from dev/test)")
    cats = sorted({r["category"].split("+")[0] for r in uniq})
    print(f"\n{'category':<24}" + "".join(f"{s:>8}" for s in ("train", "dev", "test")))
    for c in cats:
        print(f"{c:<24}" + "".join(f"{sum(r['category'].split('+')[0] == c and r['split'] == s for r in uniq):>8}"
                                   for s in ("train", "dev", "test")))


if __name__ == "__main__":
    main()
