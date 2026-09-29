"""
M5: durable journey + event persistence (Section 14's "Postgres schema").

SQLite stands in for Postgres here, same reasoning as storage.py's local
filesystem standing in for object storage and event_delivery.py's
file-per-event standing in for Kafka: a real server-backed store needs
infra this prototype has no reason to stand up, and the schema below
uses only standard SQL types/constraints -- swapping the driver for a
real Postgres connection later is a connection-string change, not a
redesign.

Closes the gap M4 left explicit: a restarted process could recover audio
(via AudioStore) but not a journey's session_token or enrolled personal
UDK, since neither was written anywhere durable. This makes a full crash
recover the whole journey, not just its audio.

M6: an optional KeyManager encrypts session_token and personal_udk_phrase
before they hit disk (Section 10: a personal UDK phrase "functions like a
credential given its role in triggering emergency response" and must
never be stored plaintext; session tokens the same). Everything else
(journey_id, user_id, status, timestamps) stays plaintext -- it's not
credential-shaped, and staying queryable/indexable matters more for it.

A 2026-09-21 audit found one more field that belonged on this list and
was missing from it: an event's transcript -- the actual spoken words of
a detected distress phrase -- is at least as sensitive as the phrase
itself, but was stored in the events table's payload blob as plain JSON.
Fixed with the same field-level encryption save_event()/load_events()
now apply, not whole-row encryption -- every other field in the payload
(decision, udk_id, confidence, timestamps) stays plaintext, same
reasoning as above.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class JourneyDB:
    def __init__(self, path: str | Path, key_manager=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key_manager = key_manager
        # check_same_thread=False only lifts sqlite3's same-thread check --
        # a single Connection object still isn't safe for concurrent use
        # from multiple threads without external serialization (found via
        # the M5 concurrent-journey load test: sporadic InterfaceError /
        # SystemError under real thread concurrency). This lock is that
        # serialization. ponytail: one global lock across all journeys,
        # not per-row -- fine at prototype load-test scale (tens of
        # threads, short transactions); a real Postgres deployment
        # wouldn't need this at all (real connection pooling per request).
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL + a busy timeout so modest concurrent access (the M5 load
        # test) doesn't trip over SQLite's default single-writer lock.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS journeys (
                journey_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_token TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                last_heartbeat_at REAL NOT NULL,
                personal_udk_id TEXT NOT NULL,
                personal_udk_phrase TEXT NOT NULL,
                personal_udk_verify_by_default INTEGER NOT NULL,
                had_trigger INTEGER NOT NULL DEFAULT 0,
                language TEXT NOT NULL DEFAULT 'en'
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                journey_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                decision TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_journey ON events(journey_id)")
        # Lazy migration for a table that already existed before had_trigger
        # was added -- CREATE TABLE IF NOT EXISTS doesn't add columns to an
        # existing table. No migration framework for a prototype; just try
        # the ALTER and ignore it if the column's already there.
        try:
            self._conn.execute("ALTER TABLE journeys ADD COLUMN had_trigger INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE journeys ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def _enc(self, value: str) -> str:
        return self._key_manager.encrypt_str(value) if self._key_manager is not None else value

    def _dec(self, value: str) -> str:
        return self._key_manager.decrypt_str(value) if self._key_manager is not None else value

    def save_journey(self, journey) -> None:
        """Upsert -- current-state storage, not an event log. Called on
        every state change worth surviving a restart (start, heartbeat,
        stop). ponytail: a full-row upsert per call, fine at
        prototype/test-journey scale; a real Postgres deployment would
        likely split hot fields (heartbeat) from cold ones (enrollment)
        if write volume ever made this a bottleneck."""
        udk = journey.personal_udk
        with self._lock:
            self._conn.execute(
                """INSERT INTO journeys
                     (journey_id, user_id, session_token, status, started_at, last_heartbeat_at,
                      personal_udk_id, personal_udk_phrase, personal_udk_verify_by_default, had_trigger, language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(journey_id) DO UPDATE SET
                     status = excluded.status,
                     last_heartbeat_at = excluded.last_heartbeat_at,
                     had_trigger = excluded.had_trigger,
                     language = excluded.language""",
                (
                    journey.journey_id,
                    journey.user_id,
                    self._enc(journey.session_token),
                    journey.status,
                    journey.started_at,
                    journey.last_heartbeat_at,
                    udk.udk_id,
                    self._enc(udk.phrase),
                    int(udk.verify_by_default),
                    int(journey.had_trigger),
                    # "" (not NULL -- the column is NOT NULL) means
                    # "still pending real detection" -- create_or_get's
                    # `recovered.get("language") or None` on the way back
                    # in correctly treats "" the same as never-detected,
                    # not as a real language.
                    getattr(journey, "language", None) or "",
                ),
            )
            self._conn.commit()

    def load_journey(self, journey_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM journeys WHERE journey_id = ?", (journey_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["session_token"] = self._dec(result["session_token"])
        result["personal_udk_phrase"] = self._dec(result["personal_udk_phrase"])
        return result

    def save_event(self, event: dict) -> None:
        """Idempotent: event_id is the primary key, so a retried save
        (Section 9's deterministic event_id) is a no-op, not a duplicate
        row -- the same guarantee storage.py's write_segment gives audio.

        Real bug found and fixed in a 2026-09-21 audit: the transcript --
        the actual spoken words of a detected distress phrase -- was
        stored in the payload blob as plain JSON, even though this same
        module already encrypts session_token/personal_udk_phrase
        specifically because Section 10 calls a personal UDK phrase
        "credential-shaped." A transcript of someone speaking it is at
        least as sensitive; it was simply the one field left out. Fixed
        with the same field-level _enc()/_dec() pattern already used for
        those two fields, not whole-row encryption -- every other field
        (decision, udk_id, confidence, timestamps) stays plaintext and
        queryable, same reasoning as the rest of this module."""
        payload = event if self._key_manager is None else {**event, "transcript": self._enc(event["transcript"])}
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO events (event_id, journey_id, timestamp, decision, payload) VALUES (?, ?, ?, ?, ?)",
                (event["event_id"], event["journey_id"], event["timestamp"], event["decision"], json.dumps(payload)),
            )
            self._conn.commit()

    def load_events(self, journey_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE journey_id = ? ORDER BY timestamp", (journey_id,)
            ).fetchall()
        events = [json.loads(r["payload"]) for r in rows]
        if self._key_manager is not None:
            for e in events:
                e["transcript"] = self._dec(e["transcript"])
        return events

    def list_journeys(self) -> list[dict]:
        """For the M6 retention sweep -- doesn't decrypt session_token or
        the personal UDK phrase, since the sweep only needs journey_id
        and started_at to decide what's expired."""
        with self._lock:
            rows = self._conn.execute("SELECT journey_id, started_at FROM journeys").fetchall()
        return [dict(r) for r in rows]

    def delete_journey(self, journey_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM journeys WHERE journey_id = ?", (journey_id,))
            self._conn.execute("DELETE FROM events WHERE journey_id = ?", (journey_id,))
            self._conn.commit()
