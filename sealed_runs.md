# Sealed test runs

Rules: new audio is split by recording into DEV (analysed freely) and SEALED before anyone runs
anything. A sealed run is only for candidates committed below, with the pass rule written first.
Only totals are shared (`evaluate_real_world.py ... --sealed`). No debugging on sealed data; a
sealed set that decides a change becomes dev afterwards.

## Run 1 -- written 2026-09-28, before running

Sealed set: the user's sealed YouTube videos (harmless audio -> false alarms per hour only).

Candidates (both with UDK_ENABLE_KWS=0, everything else = live config):
- BASE (live today): UDK_INTENT_GATE = v1 model (Kaggle dataset `mdeberta`), UDK_INTENT_THRESHOLD=0.02,
  no context / single-STT / one-word switches.
- A: UDK_INTENT_GATE = v2 model (`finetuned_v2/mDeBERTa-v3-base-mnli-xnli`, trained 2026-09-27 on
  intent_dataset_v2.csv, dev loss 0.0285), UDK_INTENT_THRESHOLD=0.05, UDK_INTENT_CONTEXT_S=20,
  UDK_INTENT_SINGLE_STT=0, UDK_INTENT_SHORT_SUPPORT=0.9.

Already known (dev / regression, not sealed): prepped danger BASE 78/84, A 78/84.
B (A + UDK_INTENT_SINGLE_STT=1) found 77/84 -> rejected by the recall rule, not run on sealed.

Pass rule for A (decided now): on the sealed ALL row, A's "Whisper + Parakeet" episodes/hour is
LOWER than BASE's, and A's "Everything live" is not higher than BASE's. Recall is already equal
(78/84). If A fails, it is not tuned on this set.

Results (2026-09-28, Kaggle, 25 sealed clips, 12.00 h, --max-min 60), ALL row, episodes/hour:

| Candidate | Whisper only | Parakeet only | Whisper + Parakeet | Scream-alone | Everything live |
|---|---|---|---|---|---|
| BASE | 1.3 | 0.9 | 2.2 | 2.4 | 4.6 |
| A    | 1.3 | 0.8 | 2.2 | 2.4 | 4.5 |

Verdict: **A FAILS** -- Whisper + Parakeet 2.2 is not lower than BASE's 2.2 (the 0.1/h drop in
Everything live is ~1 episode in 12 h). BASE stays live; A is not tuned on this set. No change was
decided by this set, so it stays sealed. Scream-alone (2.4/h) is now the largest FA source.
