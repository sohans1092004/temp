"""webhook_delivery against a real local HTTP server: signed POST, 2xx/non-2xx/unreachable,
config guards, and api.py picking it from env. python test_webhook_delivery.py"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from webhook_delivery import make_webhook_deliver, sign

SECRET = "s3cret"
received, status = [], [200]


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        received.append((body, self.headers))  # case-insensitive, like real servers
        self.send_response(status[0])
        self.end_headers()

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}/udk-alerts"
deliver = make_webhook_deliver(url, SECRET)
event = {"event_id": "abc123", "decision": "TRIGGER_ALL", "udk_id": "UDK_03", "layer": "exact"}

assert deliver(event) is True
body, headers = received[-1]
assert json.loads(body) == event
assert headers["X-UDK-Event-Id"] == "abc123"
assert headers["X-UDK-Signature"] == sign(SECRET, body)  # what the receiver recomputes
assert headers["X-UDK-Signature"] != sign("wrong", body)

status[0] = 500
assert deliver(event) is False  # stays queued for retry
srv.shutdown()
srv.server_close()
assert deliver(event) is False  # unreachable: False, never raises

for bad_url, bad_secret in (("http://example.com/hook", SECRET), ("https://example.com/hook", "")):
    try:
        make_webhook_deliver(bad_url, bad_secret)
        raise SystemExit(f"accepted bad config {bad_url!r} {bad_secret!r}")
    except ValueError:
        pass

os.environ["UDK_WEBHOOK_URL"] = url
os.environ["UDK_WEBHOOK_SECRET"] = SECRET
import api  # noqa: E402

assert api._downstream_deliver.__qualname__.startswith("make_webhook_deliver")
print("webhook_delivery: all checks passed")
