"""Fine-tuned intent model on ALL of prepped_data (6 conditions), as variations of how it
may act on today's rules. Replays the logged transcripts (evaluate_prepped_data logs)
through the current udk_engine, then applies each variant. Run from the Model folder:
    python -m intent_data.gate_variants MODEL_DIR [LOG_DIR] [LOG_PREFIX]
defaults: LOG_DIR = intent_data/logs, LOG_PREFIX = prepped_final_ (else prepped_gpu_)

Variants (p = model's P(distress) for an alert's transcript):
  rules            today, no model
  soften           p < THR turns TRIGGER_ALL into TRIGGER_VERIFY; nothing silenced
                   (current policy: models may soften, never silence)
  veto-weak@t      silence non-exact alerts with conf < 0.75 when p < t
  veto-all@t       silence ANY alert, exact included, when p < t
  veto-agree@t     veto-all, but only for words BOTH STT engines heard (see heard_by_both)
  veto-ctx@t       veto-agree, with p read on the alert + the speech segment before it
A variant is only usable if it LOSES no danger clip. Nothing here is tuned on prepped.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from finetune import FineTuned  # noqa: E402
from nli_eval import CONDS, WEAK_CONF  # noqa: E402

THR = 0.99  # finetune.py's DEV threshold for mDeBERTa (recall >= 95%)
SWEEP = [0.01, 0.05, 0.2, 0.5, 0.99]


AGREE = 80  # token_set_ratio: the other STT heard about the same words


def heard_by_both(path, text, paths) -> bool:
    """True if another STT path has a transcript matching this one. Two engines agreeing
    means the words are trustworthy; disagreement means unclear audio, where the text
    model is judging words that may never have been said, so it must not silence."""
    from rapidfuzz import fuzz

    t = text.lower()
    return any(fuzz.token_set_ratio(t, o.lower()) >= AGREE for p, segs in paths.items() if p != path for o in segs)


def alerts_after(events, p, variant, t, paths=None):
    """-> clip tier after applying a variant to its (path, event) alerts."""
    kept = []
    for path, e, ctx in events:
        weak = not e.layer.startswith("exact") and e.confidence < WEAK_CONF
        low = (p[ctx] if variant == "veto-ctx" else p[e.transcript]) < t
        if variant == "veto-all" and low or variant == "veto-weak" and weak and low:
            continue
        if variant in ("veto-agree", "veto-ctx") and low and heard_by_both(path, e.transcript, paths):
            continue
        kept.append("TRIGGER_VERIFY" if variant == "soften" and low else e.decision)
    return "TRIGGER_ALL" if "TRIGGER_ALL" in kept else ("TRIGGER_VERIFY" if kept else "nothing")


def replay_ctx(clip, R):
    """R.replay, but each alert also carries its CONTEXT: the previous speech segment +
    this segment up to the alert's sentence. A pause can split "I don't | want to go home
    now" into two segments; judged alone, the second half reads as the UDK. The previous
    segment's end punctuation is dropped: STT puts a period wherever the pause was."""
    import udk_engine
    from udks import GENERAL_UDKS

    out = []
    for path, segs in clip["paths"].items():
        engine = udk_engine.UDKEngine(GENERAL_UDKS, semantic_matcher=R.semantic())
        for i, text in enumerate(segs):
            for e in engine.decide_all(text, now_s=1000.0 + i * 0.1):
                if e.decision != "NO_ACTION":
                    j = text.find(e.transcript)
                    upto = text[:j + len(e.transcript)] if j >= 0 else text
                    # ponytail: previous segment regardless of the gap (logs have no timestamps); cap by seconds when live
                    out.append((path, e, (segs[i - 1].rstrip(" .!?") + " " if i else "") + upto))
    return out


def main() -> None:
    import scratch_replay as R

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = sys.argv[1]
    log_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "logs"
    prefix = sys.argv[3] if len(sys.argv) > 3 else next(
        (p for p in ("prepped_final_", "prepped_gpu_") if (log_dir / f"{p}normal.log").exists()), "prepped_final_")
    conds = [c for c in CONDS if (log_dir / f"{prefix}{c}.log").exists()]
    if not conds:
        sys.exit(f"no {prefix}*.log in {log_dir}")
    print(f"logs: {log_dir}/{prefix}*.log  conditions: {conds}  (missing: {sorted(set(CONDS) - set(conds))})")

    data = {c: R.parse(str(log_dir / f"{prefix}{c}.log")) for c in conds}
    replayed = {c: {n: replay_ctx(clip, R) for n, clip in clips.items() if "danger" in clip} for c, clips in data.items()}
    empty = [c for c in conds if not replayed[c]]  # no ALERT-LEVEL summary, e.g. the run crashed
    if empty:
        print(f"SKIPPED (log has no result summary -- crashed run?): {empty}")
        conds = [c for c in conds if c not in empty]
    texts = sorted({x for evs in replayed.values() for es in evs.values() for _, e, ctx in es for x in (e.transcript, ctx)})
    m = FineTuned(model_dir)
    p = dict(zip(texts, map(float, m.score(texts))))
    print(f"model: {model_dir} on {m.dev}; {len(texts)} alert transcripts scored")

    variants = [("rules", 1.0), ("soften", THR)] + [(v, t) for v in ("veto-weak", "veto-all", "veto-agree", "veto-ctx") for t in SWEEP]
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # [danger caught, danger n, controls alerting, controls n]
    full_alarm = defaultdict(lambda: [0, 0])  # variant -> [danger TRIGGER_ALL, controls TRIGGER_ALL]
    lost, silenced = defaultdict(list), defaultdict(list)
    for c, clips in replayed.items():
        for name, events in clips.items():
            danger = data[c][name]["danger"]
            base = alerts_after(events, p, "rules", 1.0)
            for v, t in variants:
                tier = base if v == "rules" else alerts_after(events, p, v, t, data[c][name]["paths"])
                k = f"{v}@{t}" if v.startswith("veto") else v
                cell = tally[k][c]
                cell[0 if danger else 2] += tier != "nothing"
                cell[1 if danger else 3] += 1
                full_alarm[k][0 if danger else 1] += tier == "TRIGGER_ALL"
                if base != "nothing" and tier == "nothing":
                    (lost if danger else silenced)[k].append(f"{c}/{name[:50]}")

    print(f"\n{'variant':<16}" + "".join(f"{c:>12}" for c in conds) + f"{'TOTAL':>14}{'full alarms':>14}   (danger caught | controls alerting)")
    for k, cells in tally.items():
        tot = [sum(cells[c][i] for c in conds) for i in range(4)]
        row = "".join(f"{f'{cells[c][0]}|{cells[c][2]}':>12}" for c in conds)
        print(f"{k:<16}{row}{f'{tot[0]}/{tot[1]}|{tot[2]}/{tot[3]}':>14}{f'{full_alarm[k][0]}|{full_alarm[k][1]}':>14}"
              + ("   LOSES DANGER" if lost[k] else ""))

    for k in tally:
        if lost[k] or silenced[k]:
            print(f"\n{k}: lost {len(lost[k])} {sorted(set(lost[k]))[:8]}")
            print(f"   silenced controls {len(silenced[k])}: {sorted({x.split('/', 1)[1] for x in silenced[k]})}")

    print("\nALERT TRANSCRIPTS of control clips (what the model saw), p = P(distress):")
    for c in conds[:1] + [x for x in conds[1:] if x == "pocket"]:
        for name, events in replayed[c].items():
            if not data[c][name]["danger"] and events:
                print(f"  [{c}] {name[:60]}")
                for path, e, ctx in events:
                    print(f"      p={p[e.transcript]:.3f} ctx p={p[ctx]:.3f}  {e.decision:<15} {e.layer:<10} {path}: {ctx!r}")


if __name__ == "__main__":
    main()
