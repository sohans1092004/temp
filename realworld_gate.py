"""Intent-model veto on the REAL non-distress recordings (YouTube real_world run + Telugu):
false alarms per hour, today's rules vs. the veto variants. Every alert here is false.
    python -m intent_data.realworld_gate MODEL_DIR
Reads intent_data/real_world/*.csv (evaluate_real_world.py's alert lists). Those runs
saved only ALERTS, not every transcript, so:
  - each alert's text is re-decided by today's rules (KWS rows dropped: KWS is off);
  - "heard by both" = the other STT raised an alert within AGREE_S seconds on a
    matching text. Same rules on the same words give the same alert, so this is close to
    the prepped replay's check -- but it can miss agreement where the other STT heard the
    words and didn't alert (then nothing is vetoed: errs toward keeping alerts).
Scream-alone alerts are listed apart: the text model can't judge them.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from finetune import FineTuned  # noqa: E402

# recorded hours per category (evaluate_real_world reports, 2026-09-25; same files, same caps)
HOURS = {"ambient": 0.26, "hard": 2.06, "speech": 2.25, "telugu": 1.18}
AGREE, AGREE_S, REFRACTORY_S = 80, 3.0, 10.0
RULE = re.compile(r"UDK (UDK_\d+) \(([\w/]+) ([\d.]+), (\w+)\)")
SWEEP = [0.01, 0.2, 0.5, 0.99]


def secs(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def episodes(times) -> int:
    n, last = 0, -1e9
    for t in sorted(times):
        n += t - last >= REFRACTORY_S
        last = t
    return n


def main() -> None:
    import scratch_replay as R
    import udk_engine
    from rapidfuzz import fuzz
    from udks import GENERAL_UDKS

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = [r for f in sorted((HERE / "real_world").glob("*.csv")) for r in csv.DictReader(open(f, encoding="utf-8"))]
    alerts, screams = [], defaultdict(list)
    for r in rows:
        if r["rule"].startswith("scream"):
            screams[r["category"]].append((r["file"], secs(r["time"])))
            continue
        m = RULE.match(r["rule"])
        if not m or m.group(2) == "kws" or not r["heard"].strip():
            continue
        engine = udk_engine.UDKEngine(GENERAL_UDKS, semantic_matcher=R.semantic())
        if any(e.decision != "NO_ACTION" for e in engine.decide_all(r["heard"], now_s=1000.0)):  # still alerts today
            alerts.append({"cat": r["category"], "file": r["file"], "t": secs(r["time"]), "stt": m.group(4),
                           "layer": m.group(2), "text": r["heard"]})
    dropped = sum(1 for r in rows if not r["rule"].startswith("scream")) - len(alerts)
    for a in alerts:
        a["agree"] = any(b["stt"] != a["stt"] and b["file"] == a["file"] and abs(b["t"] - a["t"]) <= AGREE_S
                         and fuzz.token_set_ratio(a["text"].lower(), b["text"].lower()) >= AGREE for b in alerts)
    m = FineTuned(sys.argv[1])
    texts = sorted({a["text"] for a in alerts})
    p = dict(zip(texts, map(float, m.score(texts))))
    print(f"{len(alerts)} text alerts still firing under today's rules ({dropped} old rows gone: KWS off / rules changed); "
          f"{sum(a['agree'] for a in alerts)} heard by both STTs")

    variants = [("rules", None)] + [(f"veto-all@{t}", t) for t in SWEEP] + [(f"veto-agree@{t}", t) for t in SWEEP]
    cats = sorted(HOURS)
    print(f"\ntext false alarms per hour (episodes, 10 s merge)\n{'variant':<18}" + "".join(f"{c:>10}" for c in cats) + f"{'all':>10}")
    for name, t in variants:
        kept = [a for a in alerts if t is None or not (p[a["text"]] < t and (name.startswith("veto-all") or a["agree"]))]
        by = defaultdict(list)
        for a in kept:
            by[(a["cat"], a["file"])].append(a["t"])
        per = {c: sum(episodes(v) for (cc, _), v in by.items() if cc == c) for c in cats}
        print(f"{name:<18}" + "".join(f"{per[c] / HOURS[c]:>10.1f}" for c in cats) + f"{sum(per.values()) / sum(HOURS.values()):>10.1f}")
    sc = {c: sum(episodes([t for f, t in screams[c] if f == ff]) for ff in {f for f, _ in screams[c]}) for c in cats}
    print(f"{'(scream-alone)':<18}" + "".join(f"{sc[c] / HOURS[c]:>10.1f}" for c in cats) + "   <- untouched by the text model")

    print("\nremaining at veto-agree@0.99 (most frequent first):")
    left = defaultdict(list)
    for a in alerts:
        if not (p[a["text"]] < 0.99 and a["agree"]):
            left[a["text"]].append(a)
    for text, xs in sorted(left.items(), key=lambda kv: -len(kv[1]))[:25]:
        a = xs[0]
        print(f"  {len(xs):>3}x  p={p[text]:.3f}  {'both' if a['agree'] else 'one '}  {a['layer']:<16} {a['cat']:<8} {text[:90]!r}")


if __name__ == "__main__":
    main()
