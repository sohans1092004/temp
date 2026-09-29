"""
Alert delivery to the website's server by HTTPS POST (a webhook), the
alternative to kafka_delivery.py for a platform that just wants a URL.

make_webhook_deliver() returns a callable with _downstream_deliver's
contract: takes an event dict, returns True on a 2xx and False (never
raises) otherwise, so EventDeliveryQueue keeps failed alerts for retry.

Every POST is signed: X-UDK-Signature = "sha256=" + HMAC-SHA256(secret,
raw body). The receiver MUST check it -- an unsigned alert could be forged,
and an alert here can end in a report to the authorities. X-UDK-Event-Id is
stable across retries, so the receiver can drop duplicates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def make_webhook_deliver(url: str, secret: str, timeout_s: float = 5.0) -> Callable[[dict], bool]:
    parsed = urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in LOCAL_HOSTS:
        raise ValueError("UDK_WEBHOOK_URL must be https:// (events carry transcripts)")
    if not secret:
        raise ValueError("UDK_WEBHOOK_SECRET is required: the receiver must be able to verify alerts")

    def deliver(event: dict) -> bool:
        body = json.dumps(event).encode()
        req = Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "X-UDK-Event-Id": event["event_id"],
            "X-UDK-Signature": sign(secret, body),
        })
        try:
            # ponytail: synchronous POST on the detection path, bounded by timeout_s; move to a
            # background sender if the receiver is ever slow enough to delay the stream
            with urlopen(req, timeout=timeout_s) as r:
                return 200 <= r.status < 300
        except Exception:  # HTTPError (non-2xx), URLError, timeout: all "not delivered", retried later
            return False

    return deliver
