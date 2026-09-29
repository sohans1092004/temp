"""
Batch-run every clip in real_recordings/mixed/ through the real pipeline
(reuses run_real_audio.py's process_file -- same wiring as the live
WebSocket path) and score against manifest.json's expected phrase/category
per noise `condition` (clean/noisy/muffled/overlapping).

Every clip in this set is a real distress phrase, so "an event fired"
(decision != NO_ACTION) is a true positive and "no event fired" is a
false negative. This is the noise-robustness check for the LID/STT
noise-ceiling gap open in the project (Section 17 / real_recordings gap).

Usage:
    python3 run_mixed_batch.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from run_real_audio import process_file

MIXED_DIR = Path(__file__).parent / "real_recordings" / "mixed"


def main() -> None:
    manifest = json.loads((MIXED_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest_by_stem = {Path(entry["file"]).stem: entry for entry in manifest}

    clips = sorted(MIXED_DIR.glob("*.mp3"))
    if not clips:
        print(f"No .mp3 files found in {MIXED_DIR}")
        return

    by_condition = defaultdict(lambda: [0, 0])  # condition -> [hits, total]

    for path in clips:
        entry = manifest_by_stem.get(path.stem)
        condition = entry["condition"] if entry else "unknown"
        journey = process_file(path)
        fired = len(journey.events) > 0
        by_condition[condition][1] += 1
        if fired:
            by_condition[condition][0] += 1

        status = "HIT " if fired else "MISS"
        phrase = entry["phrase"] if entry else "?"
        top = journey.events[0] if journey.events else None
        detail = f"[{top['decision']}] conf={top.get('confidence')} transcript={top.get('transcript')!r}" if top else "no event"
        print(f"{status} {path.name:28s} lang={journey.language!s:6} phrase={phrase!r:40} {detail}")

    print("\n--- Accuracy by condition ---")
    total_hits = total_n = 0
    for condition, (hits, n) in sorted(by_condition.items()):
        print(f"  {condition:12s} {hits}/{n} ({100 * hits / n:.0f}%)")
        total_hits += hits
        total_n += n
    print(f"  {'overall':12s} {total_hits}/{total_n} ({100 * total_hits / total_n:.0f}%)")


if __name__ == "__main__":
    main()
