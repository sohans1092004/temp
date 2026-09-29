"""
M4/M6's real event-bus swap. event_delivery.py's EventDeliveryQueue is
the durable local staging queue Section 6 calls for ("a local durable
queue on the event producer before Kafka") -- that part doesn't change.
What this module replaces is api.py's _downstream_deliver stub
(`lambda event: True`, "the real safety platform isn't built here") with
a real publish to a real Kafka-API-compatible broker (Redpanda here,
real Kafka in production -- kafka-python doesn't care which).

make_kafka_deliver() returns a callable matching _downstream_deliver's
exact contract: takes an event dict, returns True on confirmed delivery
and False (never raises) on failure, so EventDeliveryQueue's retry logic
in api.py keeps working unchanged regardless of which callable is plugged in.
"""

from __future__ import annotations

import json
from typing import Callable

from kafka import KafkaProducer
from kafka.errors import KafkaError


def make_kafka_deliver(bootstrap_servers: str, topic: str) -> Callable[[dict], bool]:
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda event: json.dumps(event).encode(),
        key_serializer=lambda event_id: event_id.encode(),
    )

    def deliver(event: dict) -> bool:
        try:
            future = producer.send(topic, key=event["event_id"], value=event)
            future.get(timeout=10)  # block for the broker ack -- a "delivered" claim should mean delivered
            return True
        except KafkaError:
            return False

    return deliver
