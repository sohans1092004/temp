"""
M6: audit logging (Section 10) -- every access to raw audio or a personal
UDK phrase, logged with who/when/why/outcome. Never the content itself:
Section 10 is explicit that logs typically have weaker access control and
longer retention defaults than the primary data stores, so a debug log
line that includes the actual audio or phrase is exactly the kind of
accidental-exposure path it calls out.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                journey_id TEXT NOT NULL,
                outcome TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def record(self, actor: str, action: str, journey_id: str, outcome: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (timestamp, actor, action, journey_id, outcome) VALUES (?, ?, ?, ?, ?)",
                (time.time(), actor, action, journey_id, outcome),
            )
            self._conn.commit()

    def for_journey(self, journey_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE journey_id = ? ORDER BY timestamp", (journey_id,)
            ).fetchall()
        return [dict(r) for r in rows]
