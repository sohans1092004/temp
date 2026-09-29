"""
M6: retention/deletion automation (Part 1 Section 7's default policy).

Unflagged journeys (no TRIGGER_ALL/TRIGGER_VERIFY event ever recorded)
are deleted 30 days after they started; a flagged journey gets 180 days
before deletion is even considered, matching Section 7's "flagged
journeys extend to 180 days, reviewed for deletion or legal hold at that
point rather than kept indefinitely by default."

Windows are constructor parameters, not hardcoded constants, so tests can
exercise the logic with seconds instead of waiting on real days -- the
same trick HEARTBEAT_GRACE_S already uses in api.py.

This is the sweep itself, not a scheduler: a real deployment runs it on a
cron/interval. Nothing here decides when to run, only what to do on a run.
"""

from __future__ import annotations

import time


def sweep(db, audio_store, unflagged_window_s: float, flagged_window_s: float) -> list[str]:
    """Deletes audio + DB rows for journeys past their retention window.
    Returns the journey_ids actually deleted, for observability/logging
    (the caller decides what to do with that -- this module doesn't log
    on its own, so it stays swappable behind any real deployment's
    existing logging setup)."""
    now = time.time()
    deleted = []
    for row in db.list_journeys():
        events = db.load_events(row["journey_id"])
        flagged = any(e["decision"] in ("TRIGGER_ALL", "TRIGGER_VERIFY") for e in events)
        window = flagged_window_s if flagged else unflagged_window_s
        if now - row["started_at"] > window:
            audio_store.delete_journey(row["journey_id"])
            db.delete_journey(row["journey_id"])
            deleted.append(row["journey_id"])
    return deleted
