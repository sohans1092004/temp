"""Score saved fine-tuned models on REAL false-alarm transcripts (no training).
    python -m intent_data.score_alerts path/to/real_world_alerts.csv
Every row is a harmless YouTube alert, so any row still >= threshold = a false alarm the
gate would NOT clear. Thresholds above 0.99 are shown too (dev threshold saturated at 0.99).
Writes <alerts>_scored.csv with one score column per model."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from finetune import OUT, SAVE, FineTuned  # noqa: E402

THRESHOLDS = [0.9, 0.99, 0.995, 0.999]


def main() -> None:
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    speech = [r for r in rows if r.get("heard", "").strip()]  # scream-alone rows have no words
    texts = [r["heard"] for r in speech]
    print(f"{len(rows)} alerts, {len(speech)} with words")
    dev_thr = json.loads(OUT.read_text()) if OUT.exists() else {}
    models = sorted(p for p in SAVE.iterdir() if (p / "config.json").exists()) if SAVE.exists() else []
    if not models:
        sys.exit(f"no saved models in {SAVE} (expected one folder per model, each with config.json)")
    for d in models:
        s = FineTuned(str(d)).score(texts)
        for r, v in zip(speech, s):
            r[d.name] = f"{v:.4f}"
        thr = next((v["threshold"] for k, v in dev_thr.items() if k.endswith(d.name)), None)
        print(f"\n{d.name} (dev threshold {thr}):")
        for t in THRESHOLDS:
            print(f"   thr {t}: still flagged {(s >= t).sum()}/{len(s)}")
        top = sorted(zip(s, texts), reverse=True)
        print("   highest:", [f"{v:.3f} {x[:60]}" for v, x in top[:5]])
    out = path.with_name(path.stem + "_scored.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
