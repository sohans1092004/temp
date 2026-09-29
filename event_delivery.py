"""
Outbound event delivery (Section 6's Event Engine): a local durable queue
in front of the downstream safety platform, so a crash between "detected"
and "delivered" doesn't silently drop an alert, and deterministic
event_ids (Section 9) mean a retried delivery doesn't double-alert
downstream even under at-least-once delivery.

One file per pending event -- deleting the file on successful delivery is
the "drained" signal, no manifest rewrite needed (storage.py's
write-heavy manifest pattern doesn't fit a queue that's mostly deletions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class EventDeliveryQueue:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, event_id: str) -> Path:
        return self.base_dir / f"{event_id}.json"

    def enqueue(self, event: dict) -> None:
        """Durability boundary for outbound events: on disk before
        delivery is attempted, at-least-once even across a crash here."""
        self._path(event["event_id"]).write_text(json.dumps(event))

    def pending(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.base_dir.glob("*.json"))]

    def mark_delivered(self, event_id: str) -> None:
        self._path(event_id).unlink(missing_ok=True)

    def drain(self, deliver: Callable[[dict], bool]) -> tuple[int, int]:
        """Attempt delivery of everything pending; deliver() returns True
        on success. Failures stay queued for the next drain (Section 3:
        "producer retries", "drain local queue once recovered")."""
        delivered = failed = 0
        for event in self.pending():
            if deliver(event):
                self.mark_delivered(event["event_id"])
                delivered += 1
            else:
                failed += 1
        return delivered, failed
