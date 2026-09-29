"""
M5's real Postgres swap, per db.py's own promise: "swapping the driver
for a real Postgres connection later is a connection-string change, not
a redesign." Same interface as db.py's JourneyDB (save_journey,
load_journey, save_event, load_events, list_journeys, delete_journey) --
api.py picks between them based on whether UDK_DATABASE_URL is set, not
on any code-shape difference.

INSERT ... ON CONFLICT upsert/ignore syntax is identical between SQLite
and Postgres (Postgres adopted SQLite's), so the SQL itself barely
changed from db.py -- the real differences are %s placeholders instead
of ?, and psycopg's dict_row factory instead of sqlite3.Row.
"""

from __future__ import annotations

import json
import threading

import psycopg
from psycopg.rows import dict_row


class PostgresJourneyDB:
    def __init__(self, dsn: str, key_manager=None):
        self.dsn = dsn
        self._key_manager = key_manager
        # Same reasoning as db.py's lock: one shared connection isn't
        # safe for simultaneous multi-thread use without serializing it.
        # Postgres's real value here is shared state ACROSS PROCESSES
        # (multiple API instances see the same data) -- a connection
        # pool would help intra-process throughput further, but that's
        # more machinery than this swap-demonstration needs.
        self._lock = threading.Lock()
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS journeys (
                journey_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_token TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at DOUBLE PRECISION NOT NULL,
                last_heartbeat_at DOUBLE PRECISION NOT NULL,
                personal_udk_id TEXT NOT NULL,
                personal_udk_phrase TEXT NOT NULL,
                personal_udk_verify_by_default BOOLEAN NOT NULL,
                had_trigger BOOLEAN NOT NULL DEFAULT FALSE,
                language TEXT NOT NULL DEFAULT 'en'
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                journey_id TEXT NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL,
                decision TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_journey ON events(journey_id)")
        # Postgres supports IF NOT EXISTS on ADD COLUMN directly -- no
        # try/except migration dance needed like SQLite's.
        self._conn.execute("ALTER TABLE journeys ADD COLUMN IF NOT EXISTS had_trigger BOOLEAN NOT NULL DEFAULT FALSE")
        self._conn.execute("ALTER TABLE journeys ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'en'")

    def _enc(self, value: str) -> str:
        return self._key_manager.encrypt_str(value) if self._key_manager is not None else value

    def _dec(self, value: str) -> str:
        return self._key_manager.decrypt_str(value) if self._key_manager is not None else value

    def save_journey(self, journey) -> None:
        udk = journey.personal_udk
        with self._lock:
            self._conn.execute(
                """INSERT INTO journeys
                     (journey_id, user_id, session_token, status, started_at, last_heartbeat_at,
                      personal_udk_id, personal_udk_phrase, personal_udk_verify_by_default, had_trigger, language)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (journey_id) DO UPDATE SET
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
                    udk.verify_by_default,
                    journey.had_trigger,
                    # "" (not NULL -- the column is NOT NULL) means "still
                    # pending real detection" -- see db.py's save_journey.
                    getattr(journey, "language", None) or "",
                ),
            )

    def load_journey(self, journey_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM journeys WHERE journey_id = %s", (journey_id,)).fetchone()
        if row is None:
            return None
        row["session_token"] = self._dec(row["session_token"])
        row["personal_udk_phrase"] = self._dec(row["personal_udk_phrase"])
        return row

    def save_event(self, event: dict) -> None:
        """Same field-level transcript encryption as db.py -- see that
        module's save_event docstring for the real bug this fixes."""
        payload = event if self._key_manager is None else {**event, "transcript": self._enc(event["transcript"])}
        with self._lock:
            self._conn.execute(
                """INSERT INTO events (event_id, journey_id, timestamp, decision, payload)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING""",
                (event["event_id"], event["journey_id"], event["timestamp"], event["decision"], json.dumps(payload)),
            )

    def load_events(self, journey_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE journey_id = %s ORDER BY timestamp", (journey_id,)
            ).fetchall()
        events = [json.loads(r["payload"]) for r in rows]
        if self._key_manager is not None:
            for e in events:
                e["transcript"] = self._dec(e["transcript"])
        return events

    def list_journeys(self) -> list[dict]:
        with self._lock:
            return self._conn.execute("SELECT journey_id, started_at FROM journeys").fetchall()

    def delete_journey(self, journey_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM journeys WHERE journey_id = %s", (journey_id,))
            self._conn.execute("DELETE FROM events WHERE journey_id = %s", (journey_id,))
