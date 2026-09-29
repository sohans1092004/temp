"""
M7: the metrics/SLO set from Section 11, tracked in-process.

Stdlib only (no prometheus_client): this is a JSON-exposed snapshot for a
single-process prototype, not a scrape target for a real monitoring
stack. Swapping this for real Prometheus/OTel instrumentation later is a
backend change to this same call-site shape, not a redesign -- the point
here is establishing *which numbers matter* (Section 11's own framing:
"a healthy-looking API can coexist with a completely broken detection
pipeline" -- these metrics watch the pipeline's actual work, not just
request/response health), not shipping a production observability stack.
"""

from __future__ import annotations

import threading


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.journeys_started = 0
        self.journeys_completed = 0
        self.journeys_timed_out = 0
        self.journeys_timed_out_after_trigger = 0  # went dark AFTER a UDK fired -- distinct, more urgent case
        self.chunks_received = 0
        self.sequence_gaps_detected = 0
        self.stt_latency_s: list[float] = []
        self.detection_latency_s: list[float] = []  # server_received_ts - timestamp, per event (Section 11)
        self.detections_by_decision: dict[str, int] = {}
        self.confidence_samples: list[float] = []
        self.events_delivered = 0
        self.events_delivery_failed = 0

    def journey_started(self) -> None:
        with self._lock:
            self.journeys_started += 1

    def journey_completed(self) -> None:
        with self._lock:
            self.journeys_completed += 1

    def journey_timed_out(self, after_trigger: bool = False) -> None:
        with self._lock:
            self.journeys_timed_out += 1
            if after_trigger:
                self.journeys_timed_out_after_trigger += 1

    def chunk_received(self) -> None:
        with self._lock:
            self.chunks_received += 1

    def sequence_gap_detected(self) -> None:
        with self._lock:
            self.sequence_gaps_detected += 1

    def stt_latency(self, seconds: float) -> None:
        with self._lock:
            self.stt_latency_s.append(seconds)

    def detection(self, decision: str, confidence: float, latency_s: float) -> None:
        with self._lock:
            self.detections_by_decision[decision] = self.detections_by_decision.get(decision, 0) + 1
            self.confidence_samples.append(confidence)
            self.detection_latency_s.append(latency_s)

    def event_delivered(self) -> None:
        with self._lock:
            self.events_delivered += 1

    def event_delivery_failed(self) -> None:
        with self._lock:
            self.events_delivery_failed += 1

    def snapshot(self) -> dict:
        """Section 11's table, computed on demand rather than tracked as
        running aggregates -- simpler and correct at prototype sample
        counts; a real deployment would want streaming percentiles
        instead of keeping every raw sample in memory forever."""
        with self._lock:
            total_delivery_attempts = self.events_delivered + self.events_delivery_failed
            return {
                "journeys": {
                    "started": self.journeys_started,
                    "completed": self.journeys_completed,
                    "timed_out": self.journeys_timed_out,
                    "timed_out_after_trigger": self.journeys_timed_out_after_trigger,
                },
                "audio": {
                    "chunks_received": self.chunks_received,
                    "sequence_gaps_detected": self.sequence_gaps_detected,
                },
                "stt_latency_s": {
                    "p50": _percentile(self.stt_latency_s, 0.50),
                    "p95": _percentile(self.stt_latency_s, 0.95),
                },
                "detection_latency_s": {
                    "p50": _percentile(self.detection_latency_s, 0.50),
                    "p95": _percentile(self.detection_latency_s, 0.95),
                    "worst_case": max(self.detection_latency_s) if self.detection_latency_s else None,
                },
                "detections_by_decision": dict(self.detections_by_decision),
                "confidence_distribution": {
                    "p50": _percentile(self.confidence_samples, 0.50),
                    "p05": _percentile(self.confidence_samples, 0.05),
                },
                "event_delivery_success_rate": (
                    self.events_delivered / total_delivery_attempts if total_delivery_attempts else None
                ),
            }
