"""Recall check for the intent gate ON TOP OF the current udk_engine (fuzzy fix included).
Replays the bundled prepped logs (84 danger / 96 control runs) with each saved model
silencing WEAK alerts (not exact, confidence < 0.75) that it scores below a threshold.
    python -m intent_data.gate_sweep
A threshold is only usable if 'danger LOST' is empty."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from finetune import SAVE, FineTuned  # noqa: E402
from nli_eval import CONDS, LOGS, regression  # noqa: E402

THRESHOLDS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.99]


def main() -> None:
    import scratch_replay as R

    missing = [c for c in CONDS if not (LOGS / f"prepped_gpu_{c}.log").exists()]
    if missing:
        sys.exit(f"prepped logs missing in {LOGS}: {missing}")
    replay = {c: R.parse(str(LOGS / f"prepped_gpu_{c}.log")) for c in CONDS}
    models = sorted(p for p in SAVE.iterdir() if (p / "config.json").exists()) if SAVE.exists() else []
    if not models:
        sys.exit(f"no saved models in {SAVE}")
    for d in models:
        m = FineTuned(str(d))
        print(f"\n== {d.name}")
        for thr in THRESHOLDS:
            g = regression(m, thr, replay)
            print(f"  thr {thr:<5} rules: danger {g['rules'][0]}/{g['danger_n']} controls {g['rules'][1]}/{g['control_n']} | "
                  f"rules+gate: danger {g['rules+gate'][0]}/{g['danger_n']} controls {g['rules+gate'][1]}/{g['control_n']} | "
                  f"danger LOST {g['gate_lost']}", flush=True)


if __name__ == "__main__":
    main()
