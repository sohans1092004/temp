# UDK Detection Prototype (M1)

Local prototype for `Audio -> VAD -> STT -> UDK matching -> console decision`,
per Section 14 (M1) of the review doc, built as a **v1 fast path** (Section
4/17's "v1 decision"): pretrained components only, no dataset of our own,
no custom model training — chosen because there was no time to build the
synthetic-data pipeline a trained keyword-spotting model would need.

## What's here

| File | What it does |
| --- | --- |
| `udks.py` | The 20 general UDK phrases (Section 16) + a sample personal UDK |
| `vad.py` | webrtcvad wrapper, biased toward treating ambiguous audio as speech (Section 2), with pre/post-roll padding |
| `stt.py` | STT behind one interface: `FasterWhisperSTT` (real, pretrained) and `MockSTT` (for this sandbox — see below) |
| `udk_engine.py` | Exact / fuzzy / semantic-stub matching + the Section 6 decision contract (`TRIGGER_ALL` / `TRIGGER_VERIFY` / `NO_ACTION`), including the repetition booster |
| `pipeline.py` | Wires VAD -> STT -> engine together, prints events in the Section 9 event shape |
| `test_pipeline.py` | 9 test cases (exact, fuzzy/ASR-noise, paraphrase, negatives, verify-by-default, repetition escalation, personal UDK) — all passing |

## FasterWhisperSTT — verified against real audio

Previously flagged as untested (the earlier sandbox couldn't reach
`huggingface.co`). From this environment, `FasterWhisperSTT("tiny.en")`
downloaded weights and ran successfully end-to-end: a TTS-generated WAV
saying "Help me please, call the police right now." was resampled to
16kHz mono PCM, run through real `webrtcvad` segmentation, transcribed by
the real model, and correctly matched to `UDK_01` ("Call the police") at
`TRIGGER_ALL` confidence 1.0, including the repetition-escalation path.

What's verified end-to-end now:

- Real `webrtcvad` segmentation + real `FasterWhisperSTT` transcription +
  `udk_engine` matching, on real (TTS) speech audio, via `pipeline.py`.
- All UDK matching/decision logic (`test_pipeline.py`), independent of
  which STT backend is behind it.

To use the real backend: construct `FasterWhisperSTT()` instead of
`MockSTT()` wherever a backend is built — nothing else in `pipeline.py` or
`udk_engine.py` needs to change, that's the point of the `STTBackend`
interface in `stt.py`.

## What this prototype deliberately does NOT include yet

Per Section 4/17's v1 decision:

- **No parallel keyword-spotting (KWS) path.** V1 is `STT -> matching` only,
  same shape as the original brief's pipeline. This reintroduces the
  single-point-of-failure risk Section 4 flagged (an STT outage or
  mis-transcription has no independent fallback) — an accepted, explicit
  tradeoff for speed, not an oversight.
- **Semantic matching is a fuzzy-ratio stub, not a real embedding model.**
  A real deployment would use a pretrained sentence-embedding model for the
  paraphrase case ("he's going to hurt me" -> "someone is threatening me");
  that model's weights are also on huggingface.co and hit the same network
  wall here. The stub is good enough to prove the pipeline shape, not a
  claim about real semantic-matching accuracy.
- **No speaker-embedding check on the personal UDK** (Section 5's asymmetric
  mismatch check) — same reason, and out of scope for a text-only prototype
  regardless.
- **No streaming, no journey session, no API** — that's M3, below.

## M2: personal UDK enrollment + audio storage

| File | What it does |
| --- | --- |
| `enrollment.py` | Validates a personal UDK phrase (Section 5): distinctiveness vs. ordinary conversation, and — if a sample recording is given — intelligibility via STT. Phrase text is the only artifact this produces; sample audio is used in memory only, never persisted, per Section 5's "text, not audio" rule |
| `storage.py` | `AudioStore` — chunked audio durability (Section 6): per-segment checksums, gaps recorded explicitly rather than smoothed over, ordered reconstruction by sequence number. Local-filesystem-backed for now, standing in for real object storage until M3 gives this a client/server boundary to put a durability boundary between |
| `test_m2.py` | 13 checks: enrollment validation (good/too-short/too-generic/sample-matches/sample-doesn't-match), the M2 acceptance criterion — personal UDK detected across exact/misspelled/hedged variants of the enrollment phrase, not just the exact sample — and storage (chunking, gap detection, checksum-corruption detection) |

Deliberately not built in M2 (out of scope per Section 14's roadmap, not an oversight):

- **No speaker embeddings.** Section 5 recommends this only as an optional,
  never-a-gate hardening layer; phrase-based detection alone is the
  required path, so there's nothing to add here without a concrete
  reason to.
- **No persistence/versioning of "the active personal UDK per user."**
  `enroll()` is a pure validate-and-construct step; storing it and
  deactivating an old one on re-enrollment is Postgres-backed state
  (Section 14's M5), not this milestone's job.

## M3: streaming API

| File | What it does |
| --- | --- |
| `api.py` | REST for journey lifecycle (`start`/`stop`/status/events, Section 9) + a WebSocket for audio ingestion (sequence numbers, acks, heartbeats, resumability). In-memory `JourneyStore` — Postgres-backed persistence is M5, not this milestone |
| `test_m3.py` | 21 checks: REST idempotency, WS auth rejection, sequencing/acks/heartbeat, out-of-order frame buffering catching up once a gap fills (Section 3), the tail-finalization fix below, and server-side journey timeout on a missed heartbeat |

**Transport choice:** Section 9 recommends gRPC bidi streaming by default,
falling back to WebSocket "if [the client] doesn't already use gRPC and
you want broader client support." There's no existing client stack here
to match and no reason to take on a protobuf/codegen toolchain to prove
streaming ingestion works, so WebSocket is what's built — swapping the
transport later wouldn't change anything below the ingestion boundary
(`JourneyState`/pipeline processing stays the same either way).

**Live-detection fix, found and fixed while testing:** a segment used to
only get finalized (transcribed + matched) once it was no longer the
buffer's *last* list entry — a proxy for "might still be growing" that
was wrong for a single continuous utterance with nothing spoken after it,
since that segment is *always* last. It never finalized until `stop()`
forced it through, which isn't "live" detection at all for a one-shot
phrase. Fixed in `vad.py`/`api.py`: a segment now finalizes once *more
buffered audio has arrived after its end* (`VAD.total_ms()` vs.
`seg.end_ms`) — actual proof it stopped growing, not just its position in
a list. `test_m3.py`'s `SingleFixedSegmentVAD` case exercises this
directly: a lone segment stays unfinalized while it's still the tail,
then fires on the very next frame once trailing audio proves it's closed.

**What this does and doesn't fix, in practice:** in a real journey the
mic keeps streaming ambient audio for the rest of the session, so this
closes the gap for the realistic case — a phrase followed by anything
(even silence) now finalizes live, without waiting for a second
utterance. It can't help the degenerate case where the client stops
sending frames the instant the phrase ends and nothing — not even
silence — ever arrives after it: there's no buffered proof to finalize
on, so it's still only caught at `stop()`. The real end-to-end smoke test
below hits exactly that degenerate case (a short clip with no trailing
audio), which is why it still shows detection at `stop()` rather than
mid-stream — that's the input, not a remaining bug.

**Real E2E smoke test (real webrtcvad + real `FasterWhisperSTT`, through
the actual API, not a test double):** 3.46s of real TTS speech, streamed
over the WS protocol in ~150ms frames with nothing sent after the clip
ends, correctly detected `UDK_01` ("Call the police") at `TRIGGER_ALL`
twice (repetition escalation), surfacing at `stop()` per the note above.
`stop()` took 0.261s end to end (VAD + 2x STT + matching on a
`tiny.en`/CPU model) — comfortably inside the ≤1.5s p50 / ≤4s p95 target
from Section 11. This is a smoke test on one clip, not the SLO
validation itself — that needs Section 12's real load-testing corpus
(distress-register audio, concurrent journeys, etc.), which is out of
scope for a local prototype.

## M4: reliability layer

Pulled forward per Section 14's own recommendation (ahead of M5's full
data/event infrastructure) since a version of this system without these
isn't safe to run with real journeys even in early testing. Two real gaps
were found and fixed while building this, not just documented:

| File | What it does |
| --- | --- |
| `storage.py` (hardened) | `write_segment` is now idempotent on `(journey_id, seq)` — a retried/duplicate frame (Section 3: client retries after a lost ack) is a no-op, not a second write that would double-count in `read_full_audio`. A same-seq-different-content write raises instead of silently overwriting (a real anomaly, not a retry). New `read_contiguous_prefix()` rebuilds the durable prefix for crash recovery |
| `api.py` (hardened) | `_ingest_frame` early-returns for an already-consumed seq (was previously silently re-writing the segment and leaking a `pending_frames` entry — a real bug found while building this milestone). `JourneyStore.create_or_get` rebuilds a journey's buffer/`next_expected_seq` from `AudioStore` if it's not in memory (a simulated crash: the in-memory store is wiped, the disk isn't) instead of starting over or losing what was already captured |
| `event_delivery.py` | `EventDeliveryQueue` — Section 6's Event Engine: a local durable queue in front of the downstream safety platform, so a crash between "detected" and "delivered" doesn't drop an alert, and `drain()` retries until delivery succeeds. Wired into `_run_incremental_detection`: every new event is durably queued, then delivery is attempted immediately |
| `test_m4.py` | 20 failure-injection checks against Part 1's table: duplicate/retried frames, a same-seq-different-content anomaly, a simulated crash-and-recover with backlog detection resuming correctly, downstream delivery failure + recovery with no duplicate delivery, and a real WS disconnect/reconnect mid-journey with no lost or duplicated audio |

**Ownership: this service is the sole owner of audio durability from the
moment audio reaches it.** Section 14's roadmap names "client-side local
buffering" as a separate M4 component, written for the brief's original
framing of a third-party mobile app with its own team. That's not this
integration — this service has direct access to the audio stream, so
there's no separate client-buffering layer to design a contract around or
depend on: everything from ingestion onward (buffering, retries,
durability, crash recovery) is owned right here, in `storage.py`/`api.py`,
which is exactly what this milestone hardens. A journey survives a
dropped WebSocket (state lives in `JourneyStore`, not the socket), a
reconnect resumes correctly, and a re-sent already-acked frame (e.g. if
only its ack was lost, not the frame) is safe rather than corrupting the
stored audio.

The one thing no amount of server-side buffering changes: audio that
never physically reaches this service (a true connectivity outage between
capture and here) can't be recovered from here — there's nothing to
buffer if nothing arrived. That's a network fact about the gap between
capture and ingestion, not a scoping choice, and it's the one piece that
would need *something* running before this service to hold audio during
an outage — whatever that capture layer is in this integration, not a
"separate mobile app team's problem" as the original brief assumed.

Postgres-backed durable *journey metadata* (surviving a full process
restart with the same `session_token`, not just the same audio) is M5,
below.

## M5: full data + event infrastructure

Real Postgres and Kafka both need infra this local prototype has no
reason to stand up — same call M2/M4 already made for object storage and
the outbound event queue. This milestone extends that pattern instead of
introducing a new one:

| File | What it does |
| --- | --- |
| `db.py` | `JourneyDB` — SQLite standing in for "Postgres schema" (Section 14): a `journeys` table (session_token, status, enrolled personal UDK) and an `events` table (event_id primary key, so a retried save is a no-op — the same idempotency guarantee `storage.py` gives audio). Closes the exact gap M4 left explicit: a restart now recovers the *whole* journey — same `session_token`, same personal UDK — not just its audio |
| `api.py` (extended) | `JourneyStore.create_or_get` checks the DB before assuming a journey is brand new; `_decision_to_event` now emits the **full** Section 9 schema (`user_id`, `server_received_ts`, `audio_segment_id`, `location`, `processing_version`, per-component `model_version`) instead of the M3-era trimmed version; `GET /journeys/{id}` and `GET /events` fall back to the DB when a journey isn't live in memory, so both work after a crash even before anything triggers recovery |
| `test_m5.py` | 15 checks: same-token crash recovery, durable status/events reachable via the DB alone, full schema-field presence, and a 20-concurrent-journey load smoke test (event delivery success rate, no cross-journey contamination, unique event ids under concurrency) |

**A real concurrency bug, found and fixed while building this:**
`sqlite3`'s `check_same_thread=False` only lifts the same-thread check —
it does not make one `Connection` object safe for *simultaneous* use from
multiple threads. The first run of the concurrent-journey load test threw
sporadic `InterfaceError`/`SystemError` under real thread concurrency.
Fixed with a single `threading.Lock` around every `JourneyDB` call
(`ponytail`-flagged as a global lock, not a per-row one — fine at this
load-test's scale; a real Postgres deployment wouldn't need this at all,
since it gets real per-request connection pooling for free). Confirmed
stable across repeated runs after the fix, not just a lucky single pass.

**Two honest field gaps in the "full" schema, not silently faked:**
`location` is always `null` — no location-ingestion mechanism was ever
asked for or built, and Section 1 treats it as optional/best-effort
anyway — and `model_version.kws` is always `null`, since KWS is
deliberately deferred per Section 17's v1 decision. The fields exist (so
a consumer can rely on the shape), the values are honest about what this
build doesn't produce.

**Load test scope, stated plainly:** 20 concurrent journeys is a
prototype-scale smoke test proving no cross-journey corruption and no
delivery loss under concurrent access from a single process — not the
brief's real target ("thousands of concurrent journeys," Section 20 of
the brief), which needs real load-testing infrastructure per Section 12,
out of scope for a local prototype.

## M6: security and compliance hardening

Tested against Section 10's threat-model table directly, row by row,
rather than describing controls in the abstract:

| File | What it does |
| --- | --- |
| `crypto.py` | `KeyManager` — a local versioned-key file standing in for a real KMS (same reasoning as every other local stand-in in this project), using `MultiFernet` from the `cryptography` library so key rotation (encrypt with the newest key, decrypt by trying all known ones) works the same way a real KMS's key versions do |
| `storage.py` (extended) | An optional `KeyManager` encrypts audio segments at rest. Checksums stay over plaintext deliberately — Fernet isn't deterministic, so checksumming ciphertext would make a legitimate retry of the same audio look like tampering. On-disk corruption is caught by Fernet's own authentication tag, which fails decryption outright rather than silently returning garbage |
| `db.py` (extended) | An optional `KeyManager` encrypts `session_token` and `personal_udk_phrase` — Section 10: a personal UDK phrase "functions like a credential." Everything else stays plaintext and queryable, since it isn't credential-shaped |
| `api.py` (extended) | `/v1/journeys/start` requires a scoped `X-Service-Token` header (`secrets.compare_digest`, not `==`) and is rate-limited per credential; a new `GET /v1/journeys/{id}/audio-url` does an ownership check and mints a short-lived HMAC-signed token, and `GET /v1/journeys/{id}/audio` validates it before returning decrypted audio — every access attempt audited regardless of outcome |
| `audit.py` | `AuditLog` — who/when/what/journey/outcome, in a dedicated SQLite table. Never the audio or transcript content itself (Section 10: logs have weaker access control and longer retention than primary stores, so that's exactly where accidental exposure happens) |
| `retention.py` | `sweep()` — Section 7's policy: unflagged journeys deleted 30 days after they start, flagged ones (any `TRIGGER_ALL`/`TRIGGER_VERIFY` ever recorded) get 180. A sweep function, not a scheduler — a real deployment runs it on a cron; windows are parameters so tests use seconds instead of waiting on real days |
| `test_m6.py` | 38 threat-model checks: service-token auth (missing/wrong/correct) on every real endpoint, rate-limit enforcement, encryption-at-rest verified by reading the *raw* stored bytes/DB columns directly (not just trusting the round-trip), ownership-checked signed URLs (wrong owner, tampered token, expired token, nonexistent journey), audit-log coverage without content leakage, and the retention sweep's flagged/unflagged windows |

**A real bug found and fixed while building this:** the signed-URL token
format was `f"{expires_at}.{sig}"`, and `expires_at` is a float whose own
string form contains a `.` (e.g. `1789818821.947...`) — splitting from
the left cut the timestamp in half instead of separating it from the
signature, so every valid token failed verification. Fixed by splitting
from the right instead (the hex signature itself never contains a dot).
Caught immediately by the first test run, not left in.

**A second, more serious real bug found much later (2026-09-21, during a
full failure-point audit of this project) and fixed: four real endpoints
had NO authentication at all.** Only `/v1/journeys/start` carried the
`require_service_token` dependency above -- `/v1/journeys/{id}/stop`,
journey status, `GET /v1/journeys/{id}/events` (which returns the full
transcript, confidence, and UDK match for every detection), and
`GET /v1/journeys/{id}/audio-url` had none. Anyone who knew or guessed a
`journey_id` could read a journey's complete detection history with zero
credentials. Worse, `/audio-url` didn't even check a real credential for
ownership -- it compared the recorded owner against a caller-supplied
`user_id` query parameter, a self-asserted claim, not an authenticated
one, so knowing a `journey_id` and its owner's `user_id` was enough to
obtain a real signed audio-download URL for someone else's private
safety-journey recording. `test_m6.py`'s own `client` fixture sends the
service-token header on every call it makes, which is exactly why this
had gone unnoticed by this file's own threat-model coverage until an
independent audit read the route decorators directly. **Fixed**: the
same `require_service_token` dependency was added to all four routes
(plus `GET /metrics`, found the same way); `/audio-url` now additionally
requires the journey's own `session_token` (the same per-journey secret
the WebSocket stream already required) instead of the caller-supplied
`user_id`. `test_m6.py` grew from 27 to 36 checks, explicitly covering
each route rejecting an unauthenticated call and a caller who knows the
right `user_id` but not the real `session_token`. Full regression suite:
216/216 passing.

**A third real bug found and fixed the same day, from the same audit:**
event transcripts -- the actual spoken words of a detected distress
phrase -- were stored unencrypted, inconsistent with this file's own
stated policy of encrypting `session_token`/`personal_udk_phrase`
specifically because a personal UDK "functions like a credential." A
transcript of someone speaking it is at least as sensitive; it was
simply the one field left out. Fixed with the same field-level
`_enc()`/`_dec()` pattern already used for those two fields (not
whole-row encryption -- every other field in the event payload stays
plaintext/queryable, same reasoning as before), applied to both `db.py`
and `db_postgres.py`. `test_m6.py` gained 2 more checks, verifying the
transcript isn't present in the raw stored payload and decrypts back
correctly. Full regression suite: 218/218 passing.

**What this milestone can't cover locally, named rather than skipped:**
TLS/mTLS in transit needs real certificates and a real network boundary —
out of scope for an in-process `TestClient`; run behind `uvicorn --ssl-keyfile/--ssl-certfile`
or a reverse proxy for a real deployment. A real KMS (AWS/GCP) needs cloud
infra this prototype has no reason to stand up — `crypto.py`'s local key
file is the documented stand-in, swappable later without touching
callers. `DEV_SERVICE_TOKEN` is a loudly-named placeholder — a real
deployment sets `UDK_SERVICE_TOKEN` to a properly managed secret, not the
literal string committed here.

## M7: scale and observability

Section 14 names three components here: autoscaling, the full
metrics/SLO/alert set, and edge KWS deployment. One of those three is
genuinely buildable in a local prototype; the other two are named
honestly as blocked, not faked.

| File | What it does |
| --- | --- |
| `metrics.py` | `Metrics` — the Section 11 table, tracked in-process (stdlib only, no `prometheus_client`): journeys started/completed/timed-out, chunks received, sequence gaps, STT latency, detection latency (`server_received_ts - timestamp`, i.e. audio-time to event-time, matching Section 9's ordering rule), detections by decision type, confidence distribution, event delivery success rate. Exposed as JSON at `GET /metrics`, since this is a single-process snapshot, not a real scrape target |
| `api.py` (extended) | Every metric above is recorded at its natural point (frame ingest, STT call, detection, delivery attempt, journey lifecycle transitions). New `JourneyStore.evict_finished()` frees a completed/timed-out journey's in-memory state once it's safely durable elsewhere (M5) |
| `test_m7.py` | 16 checks: metrics reflect real activity (not just request/response health, per Section 11's own framing), the eviction fix verified directly, an accelerated 200-journey soak-shaped run checking bounded memory + SLOs holding across the whole run, and a 60-concurrent-journey load test (3x M5's count) |

**A real memory leak, found and fixed while building this — not
theoretical:** `JourneyStore._journeys` never removed a journey once it
finished. In a long-running process, memory grows with *total journeys
ever handled*, not concurrent ones — exactly the kind of leak a soak test
exists to catch, and exactly why Section 12 lists soak testing as its own
layer distinct from a plain load test. Fixed with `evict_finished()`,
safe specifically because M5 made `GET /journeys` and `GET /events` fall
back to durable storage — evicting from memory doesn't lose anything a
caller can still reach. `test_m7.py` proves the count stays bounded
across 200 journeys instead of growing to 200.

**Soak test scope, stated plainly:** "SLOs hold over a multi-hour soak"
needs actual hours and real production traffic patterns — memory
fragmentation, connection pool exhaustion, and slow leaks that only
surface over real time don't show up in an accelerated same-process run.
What's built instead proves the two concrete things a real soak run would
also be checking: memory doesn't grow unboundedly with total journeys
handled, and the latency/delivery SLOs don't drift as volume increases.
That's necessary evidence, not sufficient proof multi-hour production
soak-testing would still need to close.

**Autoscaling — not built, and here's exactly why:** this whole service
is single-process, in-memory-first by construction (every M2–M6 stand-in
— `AudioStore`, `JourneyDB`, `EventDeliveryQueue`, the rate limiter — is
local-filesystem/SQLite-backed specifically because there was no reason
to stand up real infra for a prototype). That means it is *not*
horizontally scalable as built: two instances wouldn't share journey
state, the audio durability layer, or the event queue. Real autoscaling
needs those swapped for their real backends first (real Postgres, real
S3, real Kafka — each already flagged as the intended swap target in
M2/M5's own docstrings) plus a load-balancing story for the WebSocket
connections specifically (sticky routing, or moving connection state out
of the process entirely). Writing "autoscaling code" against a
single-process prototype would be theater; the actual blocker is the
infra swap, not application logic this build is missing.

**Edge KWS deployment — blocked on a decision made explicitly back in
M1, not an oversight here:** Section 17's v1 decision deferred the
parallel keyword-spotting path entirely — this build is `STT -> matching`
only, by design, for the reasons in that section (no time to build the
synthetic training-data pipeline a KWS model needs). There is no KWS
model in this codebase to deploy to the edge. Revisiting that v1 decision
(Section 17 names the paths: the training-data flywheel from Section 7c,
or a Porcupine-based KWS layer needing no dataset of its own) is what
would need to happen *before* "edge KWS deployment" becomes buildable at
all — this milestone can't skip ahead of that.

## Revisiting the v1 KWS decision: dataset-free keyword spotting

Section 17 named two fallback paths once there's time to revisit v1's
deferral: the training-data flywheel (Section 7c) or an off-the-shelf
engine needing no dataset of its own (it names Porcupine). This adds a
third: **query-by-example spoken term detection** — embed a handful of
reference clips per phrase with an existing pretrained speech encoder (no
training, no dataset), then match incoming audio against them by
similarity. `kws.py`.

Three approaches were tested against the same four real TTS clips
("Call the police", a reworded variant of it, a different UDK phrase "I
am not safe", and unrelated speech) before picking one — two were
rejected on real evidence, not assumption:

| Approach | Same phrase, reworded | Different UDK phrase | Unrelated speech | Verdict |
| --- | --- | --- | --- | --- |
| CLAP, zero-shot text-audio | — | similarity 0.119 | similarity **0.174** (higher than the real match's text, 0.197) | Rejected — CLAP's audio-text space is tuned for broad scene/caption matching ("a dog barking"), not fine-grained spoken-word content; ~2% margin isn't usable |
| CLAP, audio-to-audio (mean-pooled) | similarity 0.923 | similarity **0.968** (higher than same-phrase) | similarity 0.887 | Rejected — got the ordering backwards |
| Wav2Vec2 (mean-pooled) | similarity 0.972 | similarity 0.954 | similarity 0.949 | Right ordering, but ~2% margins — mean-pooling over the whole utterance collapses timing and lets voice/prosody dominate |
| **Wav2Vec2 + DTW over frame-level features** | **distance 0.219** | distance 0.271 | distance 0.365 | **Used.** Real, usable separation — DTW preserves *when* things happen instead of collapsing to one vector |

`kws.py`'s `Wav2Vec2DTWSpotter` implements the winning approach:
`enroll(phrase_id, pcm)` embeds a reference clip (TTS-generated is fine —
Section 17's own recommended bootstrap for narrow-vocabulary KWS with no
natural corpus); `spot(pcm)` embeds incoming audio and returns the
closest enrolled phrase if its DTW distance clears `DEFAULT_MATCH_THRESHOLD`.
`MockKWS` is the test double, same pattern as `stt.py`'s `MockSTT`.

### Threshold calibration against a real measured corpus

`DEFAULT_MATCH_THRESHOLD` started at 0.30 — picked from the 4-clip table
above, explicitly flagged as a starting point needing calibration.
`calibrate_kws_threshold.py` does that calibration for real: **80
positive clips** (each of the 20 general UDKs, spoken as both the exact
phrase and a natural paraphrase, in two TTS voices — Zira and Hazel —
that are deliberately *not* the voice (David) the reference bank itself
was enrolled with, so this tests real cross-voice generalization, not
self-matching) and **40 negative clips** (20 ordinary-conversation
phrases across the same two voices).

The result reframed the whole question — the old 0.30 default wasn't
just unmeasured, it was badly wrong:

| Threshold | Recall | False-positive rate |
| --- | --- | --- |
| 0.15 | 91.9% | 0.0% |
| **0.18 (picked)** | **94.6%** | **7.5%** |
| 0.20 | 97.3% | 20.0% |
| 0.22 | 98.6% | 27.5% |
| 0.25 | 100.0% | 50.0% |
| 0.30 (the old default) | 100.0% | **97.5%** |

At the old threshold, 97.5% of ordinary conversation clips would have
matched *some* UDK reference — the KWS layer would have been noise, not
signal, a lone false match on nearly every utterance. Picked **0.18**
over the zero-false-positive option (0.15): this corpus is all calm TTS
speech, and real distress delivery (shouting, panic, out of breath) is
expected to deviate *further* from a calm reference clip than a calm
paraphrase does — leaving a little recall margin for that gap this
corpus can't test seemed better than optimizing purely for today's
measured FPR. A false match still only costs a confirmation prompt
(`KWS_MATCH_CONFIDENCE` lands in `TRIGGER_VERIFY` range, not
`TRIGGER_ALL`, unless STT agrees or it repeats), matching Section 1's
"bias toward false positives" guidance — but 7.5% isn't free, which is
exactly why this needed measuring rather than guessing generously.

**A limitation calibration surfaced that no threshold fixes:** 6 of the
80 positive clips were closest to a *different real UDK phrase*, not
their own — "I don't feel safe at all" (paraphrasing UDK_02, "I'm not
safe") landed closer to UDK_14 ("I don't feel safe here") than to its own
reference; similarly UDK_07/UDK_04 and UDK_08/UDK_17. These phrase pairs
are genuinely similar to *each other*, not to ordinary speech — the
practical effect is "the right kind of alert, wrong specific phrase_id
logged," not a missed or false detection mixed in with normal
conversation. Worth knowing, not something this module tries to
disambiguate further.

**Re-verified after retightening:** the same real end-to-end proof below
(real speech, real STT outage, live API) still detects correctly at
0.18 — with one honest difference from the pre-calibration run: at the
old looser 0.30, two overlapping segments both cleared the threshold,
escalating to `TRIGGER_ALL` via repetition. At the properly-calibrated
0.18, only one segment clears it (`TRIGGER_VERIFY`) — real speech sits
closer to the boundary than the synthetic corpus suggested. That's the
calibration doing its job, not a regression; it's exactly why this
needed real measurement instead of a number that merely looked
reasonable.

**Still not Section 12's full corpus, stated plainly:** no real distress
delivery (shouting, crying, out of breath), no muffling/pocket
simulation, no real human speakers, no background noise, English only,
one TTS engine. This is a real, measured, order-of-magnitude-better
foundation than 4 eyeballed examples — not a claim that Section 12's
testing layer is now complete.

**Dependencies:** `torch` + `transformers` — install torch first from the
CPU wheel index (`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
then `pip install transformers`. This is a much heavier install than
anything else in this project (a multi-GB first-time model download,
slower CPU inference) — a real cost, stated plainly, not hidden.

### Wired into the live pipeline (Section 4/6's Approach E, fully built)

| File | What it does |
| --- | --- |
| `udk_engine.py` (extended) | `UDKEngine.decide()` takes an optional `kws_match` alongside the transcript. The two branches are fused: best-of-two by confidence, plus an **agreement bonus** — if STT and KWS independently land on the same UDK, that's cross-validation from two unrelated mechanisms, strong enough to force `TRIGGER_ALL` on its own, including overriding a verify-by-default phrase (the same override repetition already gets) |
| `pipeline.py` / `api.py` (extended) | Both call `kws.spot(seg.pcm)` alongside `stt.transcribe(seg.pcm)` on every segment. **Real bug fixed while wiring this in:** both files had `if not transcript.text: continue`, which skipped calling the engine entirely on an empty transcript — that would have silently defeated KWS-only detection the instant STT actually failed, the exact scenario Section 4 cares about. Now the skip only happens if *both* signals are empty |
| `JourneyState.kws` | Optional per-journey KWS backend (`None` = disabled). `api._kws_backend` loads a saved reference bank at startup, but **only behind `UDK_ENABLE_KWS=1`** — loading `torch`+`wav2vec2` takes 10+ seconds and adds real per-segment inference cost, and none of M1–M7's existing tests should silently get slower or start running real model inference against synthetic test PCM just because a reference file happens to exist on disk. Off by default; every existing test still runs exactly as fast as before |
| `enroll_kws_references.py` | Windows-only setup script (System.Speech via PowerShell, same technique as the M1 STT verification): synthesizes a TTS reference clip per general UDK phrase, saves the resulting embeddings via `Wav2Vec2DTWSpotter.save_references()`. Run once; the saved `.npz` loads on any OS with no TTS dependency at runtime. Already run for this checkout — `kws_references.npz` has all 20 general UDKs enrolled |
| `calibrate_kws_threshold.py` | Measures recall/false-positive-rate across a threshold sweep on a real corpus (80 positive + 40 negative clips, cross-voice) — see the calibration section below. Not part of the routine test suite (Windows-only TTS + a full threshold sweep is a deliberate, occasional re-calibration tool, same category as `enroll_kws_references.py`, not a fast repeatable check) |
| `enrollment.py` (extended) | `enroll()` takes an optional `kws` backend — if a personal UDK's real enrollment sample audio is provided, that same clip becomes its KWS reference too, extending Approach E to personal UDKs, not just the general 20 |
| `test_kws.py` | Grew from 4 to 16 checks: the fusion logic (KWS-only detection, agreement escalation, disagreement doesn't falsely corroborate, unrecognized phrase_ids are ignored), enrollment wiring, and — the one that actually matters — a live-API test proving a real detection fires through the full `_ingest_frame`/`_run_incremental_detection` pipeline while STT returns nothing for every segment |

**A real threshold finding from testing at scale, not swept under the
rug:** with only 2 reference phrases enrolled, an unrelated clip was
cleanly rejected. With the full 20-phrase bank, an unrelated clip *did*
cross `DEFAULT_MATCH_THRESHOLD` once (matched "Get away from me" at
distance 0.294, just under 0.30) — more references means more chances
for an unrelated utterance to land close to *one* of them by chance. Not
tightened in response to one example, which would be overfitting to n=1,
exactly what Section 12 warns against. It doesn't need to be: a lone KWS
match lands at `KWS_MATCH_CONFIDENCE` (0.65), which is `TRIGGER_VERIFY`
range, not `TRIGGER_ALL`, unless STT agrees or it repeats — so a false
match here costs a confirmation prompt, not a false alarm. Real
calibration still needs Section 12's actual corpus.

**Final real end-to-end proof, not just unit tests:** real `webrtcvad` +
real `FasterWhisperSTT` (deliberately made to return an empty transcript
for every segment, simulating a total STT outage) + the real, saved
20-phrase reference bank, streamed through the actual live WebSocket API
with `UDK_ENABLE_KWS=1`. Real 3.46s speech clip ("Help me please, call
the police right now."), STT completely dark the whole time — KWS alone
correctly detected `UDK_01` ("Call the police") via the live pipeline,
`transcript: ""` on the resulting event, honestly — the detection came
entirely from audio, not text. This is the actual single-point-of-failure
removal Section 4 was about, demonstrated working, not just implemented.
(Re-run after threshold calibration below: at 0.18 exactly one segment
clears the threshold, `TRIGGER_VERIFY`, versus two at the old 0.30 — see
the calibration section for why that's expected, not a regression.)

**To turn it on:** `UDK_ENABLE_KWS=1 uvicorn api:app` (or set the env var
however your deployment does that). Nothing else changes — the reference
bank ships in this checkout, and every journey created from then on gets
the parallel KWS path automatically.

## Real semantic matching (Section 4/17's other deferred piece)

`udk_engine.py`'s layer 3 (paraphrase matching, e.g. "he's going to hurt
me" catching "someone is threatening me") was a rapidfuzz
`token_set_ratio` stub, explicitly flagged as standing in for "a real
deployment['s]... pretrained sentence-embedding model" once
huggingface.co was reachable. It's reachable now, so `semantic.py` adds
the real thing: `SentenceTransformerSemanticMatcher`, optional and
injectable (`UDKEngine(udks, semantic_matcher=...)`), falling back to the
exact same stub as before when none is given — nothing about the default
behavior changed.

**Three models compared on 5 real phrase pairs before picking one**
(smaller-scale than `kws.py`'s 120-clip calibration, but a real
comparison, not a blind swap):

| Pair | Stub | multilingual-MiniLM-L12 | all-MiniLM-L6 | all-mpnet-base |
| --- | --- | --- | --- | --- |
| "he's going to hurt me" / "someone is threatening me" (Section 4's own example) | 0.435 | 0.372 | 0.402 | 0.375 |
| "call the police" / "someone is trying to hurt me" (different UDKs — should be dissimilar) | 0.233 | 0.257 | 0.388 | 0.508 |
| "I'm not safe" / "I don't feel safe here" (near-duplicate UDKs — should be similar) | 0.529 | **0.761** | 0.727 | 0.746 |
| "call the police" / "what a lovely day for a walk" (unrelated — should be dissimilar) | 0.341 | 0.148 | 0.068 | **-0.014** |
| "get away from me" / "please leave me alone right now" (paraphrase) | 0.426 | 0.593 | 0.529 | 0.532 |

**Picked `paraphrase-multilingual-MiniLM-L12-v2`** — best overall balance
(cleanest separation between the two "should be dissimilar" cases while
still recognizing paraphrase and near-duplicate pairs well), and it
satisfies Section 17's explicit "multilingual" requirement, which the two
English-only alternatives don't.

**Stated plainly, not glossed over:** every model tested, including the
one picked, scores Section 4's own flagship example *lower* than the old
stub did (0.372-0.402 vs. 0.435), and none of the four clears the
existing 0.55 semantic threshold on that specific pair. This looks like a
genuinely hard case for sentence-embedding cosine similarity at this
model size — related in meaning but different in framing and structure.
The existing 0.55 threshold is kept as-is for now: real true-positive
pairs (0.59-0.76) sit comfortably above it and real true-negative pairs
(0.15-0.26) sit comfortably below it, so there's no evidence it needs
changing, but it hasn't been swept the way `kws.py`'s threshold was.

**Dependency and opt-in, same pattern as KWS:** `sentence-transformers`
(pulls in `torch`, already needed for KWS). Gated behind
`UDK_ENABLE_SEMANTIC=1` in `api.py` — loading a second real model adds
real seconds and real per-match inference cost, and none of the existing
tests should pay that just because `torch` is already installed for KWS.
Verified working via the opt-in path against a real phrase pair this
session; `test_semantic.py` covers the wiring/fallback logic fast and
portably via a fake backend (5 checks).

## Real infrastructure: Postgres, S3 (MinIO), Kafka (Redpanda)

Every local stand-in in this project (`db.py`'s SQLite, `storage.py`'s
filesystem, `event_delivery.py`'s file-per-event queue) has said the same
thing from the start: "swapping this for the real backend is a
connection-string change, not a redesign." `docker-compose.yml` +
three new modules make good on that, verified against real running
containers, not just written against their APIs on faith.

```bash
docker compose up -d   # postgres:5432, minio:9000/9001, redpanda:9092
```

| File | What it does |
| --- | --- |
| `db_postgres.py` | `PostgresJourneyDB` — identical interface to `db.py`'s `JourneyDB`. The upsert/ignore SQL barely changed (Postgres adopted SQLite's `ON CONFLICT` syntax); the real differences are `%s` placeholders and `psycopg`'s `dict_row` factory |
| `storage_s3.py` | `S3AudioStore` — identical interface to `storage.py`'s `AudioStore`, backed by any S3-compatible endpoint via `boto3` (MinIO here, real AWS S3 in production — the client doesn't care which). Same layout as the local version: `{journey_id}/segments/{seq:06d}.pcm` + `{journey_id}/manifest.json`, just as S3 keys instead of filesystem paths |
| `kafka_delivery.py` | `make_kafka_deliver(bootstrap_servers, topic)` — returns a callable matching `_downstream_deliver`'s exact contract (event in, `True`/`False` out, never raises), backed by a real `kafka-python` producer against any Kafka-API-compatible broker. `event_delivery.py`'s durable local queue is unchanged — Section 6 wants that staged locally *before* the bus regardless of what the bus is |

**Redpanda, not Apache Kafka + ZooKeeper:** same wire protocol, same
client libraries, one container instead of a multi-node coordination
cluster. A real production Kafka deployment is heavier ops than this
prototype needs to model to prove the swap works.

**Opt-in via env vars, same pattern as KWS/semantic — nothing changes
unless you set them:**

```bash
UDK_DATABASE_URL=postgresql://udk:udk_dev_password@localhost:5432/udk
UDK_S3_ENDPOINT=http://localhost:9000
UDK_S3_BUCKET=udk-audio
UDK_S3_ACCESS_KEY=udk_minio
UDK_S3_SECRET_KEY=udk_dev_password
UDK_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
UDK_KAFKA_TOPIC=udk-events
```

Each loader (`_load_db`/`_load_audio_store`/`_load_downstream_deliver` in
`api.py`) falls back to the local stand-in if the real backend fails to
connect at startup — a real deployment that sets these env vars should
fix a broken database, not have this silently paper over it forever, but
a *local dev run* with the containers not up shouldn't refuse to start
either.

**Verified with real containers, not just written against their APIs:**
each backend individually (Postgres round-trip including the encryption
layer; MinIO round-trip including a duplicate-write idempotency check and
a raw-bytes-are-encrypted check; Kafka publish-then-consume-back through
the real broker) — then all three *together*, through the actual live
API: started a journey, streamed audio, got a real detection, stopped it,
then reconnected to Postgres, MinIO, and Kafka with **fresh, independent
clients** (not the app's own objects) and confirmed the journey row, the
audio objects, and the published event were all really there. That last
check matters — it's the difference between "the code calls the client
library" and "the data actually landed on infrastructure outside the
process."

**Not built:** connection pooling (one lock-guarded connection per
backend, same as the SQLite/local versions — real throughput tuning is a
separate concern from proving the swap works), Kafka topic
partitioning/consumer-group design (this prototype has no consumer of
its own — "the real safety platform isn't built here" still stands, just
for the *consuming* side now), and TLS to any of these three backends
(local dev containers, not a production security review — Section 10's
"never simplify away security measures" is about this system's own API,
not about hardening someone else's infra it happens to now talk to).

## Improvement attempt: multiple KWS reference voices (result: didn't work)

Tried after seeing the 40%/50% distress-sim/overlapping recall numbers
below: `enroll_kws_references.py` now enrolls all 3 installed voices per
general UDK phrase instead of 1, and `kws.py`'s `Wav2Vec2DTWSpotter`
supports several references per phrase (`spot()` takes the best match
across all of them; `clear(phrase_id)` supports re-enrollment per Section
5). Measured, not assumed to help:

| Metric | 1-voice bank | 3-voice bank |
| --- | --- | --- |
| KWS-only recall (cross-voice paraphrase corpus) | 94.6% | 93.9% (unchanged) |
| KWS-only false-positive rate | 7.5% | **15.0%** (roughly doubled) |
| Whole-pipeline recall, distress-sim | 40% | 45% (within noise, n=20) |
| Whole-pipeline recall, overlapping | 50% | 50% (unchanged) |
| Whole-pipeline false-positive rate | 15%/10% | 15%/10% (unchanged) |

**It didn't fix what it was tried for.** More reference voices doesn't
help when the *query* audio itself — pitch/tempo-shifted, or mixed with
a second speaker — is too distorted for wav2vec2's embedding to
represent well in the first place. The bottleneck for those two
conditions is the audio signal, not reference-voice coverage. Kept the
multi-voice bank anyway, since it costs nothing at the whole-pipeline
level and has real (if untested here) value for voice/accent robustness
on clean-ish speech, which Section 12 separately asks for — just not
sold as a fix for the degraded-condition gap. `DEFAULT_MATCH_THRESHOLD`
stays at 0.18 despite the isolated KWS-layer false-positive increase,
since it didn't propagate to a worse end-to-end result in this corpus.

## Section 12's test corpus, built and measured for real

No real distress recordings exist to test against — Section 17 already
established that (it's the same reason the KWS dataset didn't exist).
The answer is the same technique Section 17 recommends for building KWS
training data, applied here to a whole-pipeline evaluation corpus
instead: TTS baselines run through real signal augmentation
(`audio_augment.py`: pitch-shift + time-stretch as a stressed-voice
proxy, low-pass filtering for pocket/bag muffling, synthesized noise
mixing, and genuine overlapping speech — two real clips summed, not
simulated), then every clip run through the **real, complete pipeline**
(`evaluate_pipeline_corpus.py`: real `webrtcvad` → real
`FasterWhisperSTT` → real `Wav2Vec2DTWSpotter` KWS → real `UDKEngine`
fusion) — not KWS in isolation the way `calibrate_kws_threshold.py` was.

**Measured recall by condition** (20 general UDKs each):

| Condition | Recall | `TRIGGER_ALL` | `TRIGGER_VERIFY` |
| --- | --- | --- | --- |
| Baseline (clean TTS) | 100% | 20 | 0 |
| Muffled (pocket/bag sim) | 100% | 19 | 1 |
| Noisy (10dB traffic-like) | 100% | 20 | 0 |
| Pitch/tempo-distorted (originally called "distress-sim" — see below, this label was wrong) | 40-45% | 4 | 9 |
| Overlapping speech (real, two clips summed) | 50% | 3 | 9 |

**Overlapping speech is a real, confirmed weak point** — exactly what
Section 2 warned about ("the hardest case for both VAD and STT"), now
with real numbers instead of a prediction.

**The "distress-sim" number above turned out to be measuring the wrong
thing, and validating it against real data caught this — not a
theoretical concern, a real correction.** `validate_distress_with_tess.py`
downloads the TESS dataset (real human recordings — 2 actresses, the
same fixed word set spoken in 7 real emotions each, so the same word by
the same speaker can be compared across neutral vs. fear vs. angry) and
runs real `FasterWhisperSTT` on all three. Real result:

| Emotion | Real STT word-recognition accuracy |
| --- | --- |
| Neutral | 80.0% |
| Fear (real, acted) | 76.7% |
| Angry (real, acted) | 76.7% |

**Real fear/anger delivery cost STT only ~3-4 accuracy points, not the
~55-60 point collapse the synthetic "distress-sim" condition showed.**
Measuring the *actual* pitch/tempo shift in that real data
(`measure_real_distress_shift.py`) ruled out "wrong parameters" as the
explanation: real fear shifts pitch up by **+8.05 semitones** (more than
the +3 this project's synthetic version used) and speeds up **1.28x**
(almost exactly the 1.3x used) — a *larger* real shift than what was
synthesized, with barely any accuracy cost. The problem was never the
magnitude, it's the technique: a real vocal tract at a higher pitch still
produces clean, coherent harmonics; a phase-vocoder pitch-shift smears
them into artifacts a real higher-pitched voice never has. That's not a
calibration fix, it's a wrong-technique finding — **naive digital
pitch/tempo shifting is not a valid distress-register proxy for testing
STT robustness, at any parameter setting.** The condition has been
relabeled `pitch_tempo_distorted` in `evaluate_pipeline_corpus.py` and
kept only as a generic "how does the pipeline handle heavily distorted
audio" check, not a distress simulation.

**What this means for "how accurate is the system" (the actual question
this was chasing):** the honest revised answer is that real emotional
delivery is likely *much* closer to the 100% baseline number than the
40-45% synthetic estimate suggested — genuinely good news, found by
checking a synthetic proxy against real data rather than trusting it.
Caveat that still matters: TESS is *acted* fear/anger read from a script,
not genuine panic under real threat (interrupted speech, screaming,
being out of breath) — this narrows the uncertainty a lot, it doesn't
fully close it.

**Checked against a second, independent real dataset
(`validate_distress_with_ravdess.py`/`measure_real_distress_shift_ravdess.py`)
— not a re-run of the same evidence.** TESS has a real limitation: 2
speakers, both older Canadian actresses, single words. RAVDESS is a
different dataset entirely — 24 gender-balanced professional actors (used
8 here), two full carrier sentences ("Kids are talking by the door." /
"Dogs are sitting by the door.") instead of single words, same matched-
comparison design (same actor, same sentence, same repetition, only
emotion differs). Real result, n=32 clips per emotion:

| Emotion | Real STT word-recognition accuracy (RAVDESS) | Real pitch shift vs. neutral | Real tempo vs. neutral |
| --- | --- | --- | --- |
| Neutral | 96.9% | — | — |
| Fear (real, acted) | **100.0%** | +5.65 semitones | 1.00x (no change) |
| Angry (real, acted) | **100.0%** | +3.26 semitones | 0.92x (slightly slower) |

**This doesn't just replicate the TESS finding, it strengthens it: real
fear/angry delivery scored *higher* than neutral here, not lower** (the
3 neutral misses were all normal transcription noise, e.g. "kibs" for
"kids" — nothing about calm delivery being clearer). The real pitch
shifts differ from TESS's (RAVDESS actors didn't universally speed up the
way TESS's did — angry was actually 0.92x, slightly slower), which is
expected given different actors and material, but the conclusion doesn't
change: real emotional pitch/tempo shifts of this magnitude, on two
separate real datasets with different speakers and different material,
cost little-to-nothing in STT accuracy — sometimes nothing at all. The
`pitch_tempo_distorted` condition's 40-45%-to-65% collapse remains a
digital-pitch-shift-artifact finding, not a distress-delivery finding,
now confirmed twice independently rather than once.

**A real, unrelated bug this surfaced, found and fixed:** measuring the
negative set's false-positive rate turned up something the KWS
calibration didn't touch — **45% of ordinary conversation false-
positived**, and every single one of them through `udk_engine.py`'s
*fuzzy* layer (0 through exact, semantic, or KWS). `rapidfuzz`'s
`partial_ratio` inflates on short UDK phrases against everyday sentences
that merely share a few common words ("Call me later" hit 0.67 against
no real UDK relationship). This is the exact same class of problem
KWS's original threshold had — a real signal never measured against a
real negative corpus — just in a different layer. Fixed the same way:
applied the 0.9× "weaker signal" discount the semantic stub already had
to the fuzzy layer too, chosen specifically because it preserves the
existing, deliberately-designed ASR-noise test case ("call the plice
right now" → 0.97 raw → 0.873 discounted, still clears
`TRIGGER_ALL_THRESHOLD`) while pushing most of the measured false
positives back below `TRIGGER_VERIFY_THRESHOLD`. Re-measured after the
fix:

| Condition | False-positive rate (before) | False-positive rate (after) |
| --- | --- | --- |
| Baseline negatives | 45% | **15%** |
| Degraded negatives (muffled+noisy) | 35-40% | **10%** |

Not zero — 3 of 20 negatives still false-positive, and they're the more
genuinely-arguable cases (e.g. "I need to go now" vs. UDK_05 "I need help
right now" — real, if coincidental, lexical/phonetic overlap), not
coincidental few-shared-word noise like before. This is a real,
measured, ~70% reduction — not a claim the fuzzy layer is now perfectly
calibrated; a proper sweep (the same treatment `calibrate_kws_threshold.py`
gave KWS) is the obvious next step if this matters more than a
first-pass fix.

**Explicitly not covered, stated plainly rather than faked:** genuine
panic under real threat (TESS's real speakers are *acting* fear/anger
from a script, not in real danger), real environmental noise recordings
(synthesized band-limited noise, not field recordings), and
multilingual/code-switching speech (every installed TTS voice is
English, and TESS is English too). This is a real, measured,
order-of-magnitude improvement over "0 clips tested under degraded
conditions, 0 validated against real human speech" — not a claim that
Section 12's corpus requirement is now fully satisfied.

## A real false-alarm scenario, found by testing, then fixed with real verification

Tested directly: a group of friends teasing each other, one says "I will
call the police on you." Real result before any fix: this fired the
**exact-match layer** (confidence 1.0) against UDK_01 ("Call the
police"), because "call the police" is a literal substring of the joke
— and exact match is the one layer that skips straight to `TRIGGER_ALL`
with zero confirmation step. A joke fired the system's most severe
alert.

Three changes, each verified with real evidence, address the underlying
concerns this surfaced:

**1. Multiple UDKs firing within a window escalate confidence, not just
repetition of the same one.** `udk_engine.py`'s `DISTINCT_UDK_ESCALATION_COUNT`:
two *different* UDKs matching within `REPEAT_WINDOW_S` now force
`TRIGGER_ALL`, the same way the same UDK repeating already did — two
independent phrases both indicating distress is harder to explain away
than one ambiguous match. Verified: a verify-by-default phrase alone
stays `TRIGGER_VERIFY`; a second, *different* UDK within the window
escalates it to `TRIGGER_ALL` even though neither individually repeated
(`test_pipeline.py`'s distinct-escalation case).

**2. Going dark right after a trigger gets a tighter, distinct timeout.**
Previously every silent journey timed out the same way regardless of
*why* it went silent. `JourneyState.had_trigger` now tracks whether any
`TRIGGER_VERIFY`/`TRIGGER_ALL` fired; if the heartbeat then lapses, the
grace window tightens (`POST_TRIGGER_HEARTBEAT_GRACE_S`, 10s vs. the
ordinary 20s) and the journey gets a distinct status,
`TIMED_OUT_AFTER_TRIGGER`, instead of plain `TIMED_OUT` — so the
downstream platform (which owns dispatch, per Section 7a) can tell
"session ended normally" from "went dark right when it mattered" and
react accordingly. Persisted through `db.py`/`db_postgres.py` (a lazy
`ALTER TABLE ADD COLUMN`, no migration framework for a prototype) so it
survives a crash too; verified against both the local SQLite path and a
real, already-populated Postgres container. `metrics.py` tracks this
status as its own counter, not lumped into the ordinary timeout number.

**3. The actual phrase that caused the false alarm was revised, using
real measurement, not a guess.** UDK_01 was "Call the police" — natural
under real distress, but also short and generic enough to appear as a
literal substring inside a joke. Before settling on a replacement,
candidate phrases were measured against *two* real cases simultaneously:
does it still catch the real ASR-noise regression case ("call the plice
right now") and does it stop being a literal substring of the joke case?
The first candidate tried ("Somebody call the police, please") solved
the joke problem but measurably broke the ASR-noise case — its own
fuzzy score dropped enough that an unrelated UDK won instead, an error
this project's own testing discipline caught before committing to it.
The phrase actually used, **"Call the police, I need help,"** keeps
fuzzy-match strength close to the original (0.837 vs. 0.966 raw
`partial_ratio`) while no longer being a substring of the joke. Real,
accepted side effect: a single noisy instance of the *longer* phrase now
lands at `TRIGGER_VERIFY` instead of `TRIGGER_ALL` — not a regression,
the tiered response (plus the two mechanisms above) is exactly what's
supposed to absorb that. UDK_20 ("I need to get out of here") got the
same `verify_by_default` treatment already used for UDK_14/UDK_18, since
it measurably false-positived against ordinary conversation ("Let's get
out of here") in the exact same way.

**Re-measured after all three changes**, not just asserted to work: the
joke phrase itself now lands at `TRIGGER_VERIFY` (confidence 0.695 via
the semantic layer) instead of an unverified `TRIGGER_ALL`. The full
corpus (`evaluate_pipeline_corpus.py`) was re-run end to end: baseline
recall unchanged at 100%, `pitch_tempo_distorted`/overlapping unchanged
at 45%/50% (this fix doesn't touch those, and wasn't expected to), and —
the specific thing being fixed — UDK_01 no longer appears anywhere in
the measured false-positive list at all, where it previously did. The
overall false-positive rate held at 15%/10%, not worse, not better on
average, but the *specific* failure mode demonstrated in this session is
gone.

**What this doesn't solve, stated honestly:** phrase revision narrows
the *accidental* substring-overlap problem, it doesn't eliminate the
deeper one — the system still has no concept of who's being addressed or
whether something was said sarcastically. A sufficiently deliberate, in-
character joke using the *exact* longer phrase would still risk a
confirmable `TRIGGER_VERIFY`, same as any lexically similar phrase would.
That's the accepted ceiling of a lexical/audio-similarity system without
real natural-language-understanding of intent — the tiered response and
escalation rules exist precisely because that ceiling can't be
engineered away by phrase choice alone.

## Closing the two remaining honest gaps: overlapping speech and pitch/tempo distortion

After the false-alarm fixes above, two gaps were still openly unsolved:
overlapping-speech recall stuck at 50%, and `pitch_tempo_distorted` recall
stuck at 40-45% (already known by this point to likely overstate real
distress, per the TESS finding above, but still a real distortion-
robustness number that hadn't moved). Both got a real, measured fix —
neither is "solved" to 100%, and both real tradeoffs are stated below,
not hidden.

**Overlapping speech: SepFormer separation, as a NO_ACTION fallback only
(`separation.py`).** The 10 real overlap failures from the corpus were
reproduced directly (not assumed), then re-run through a pretrained
2-speaker separator (`speechbrain/sepformer-wsj02mix`, Section 17's
"existing model" discipline again, not a trained-from-scratch model):
mixed audio is split into two estimated per-speaker streams, and each is
re-run through STT+KWS+`UDKEngine`. Real result:

| Metric | Before | After |
| --- | --- | --- |
| Overlapping-speech recall | 50% | **65%** (3/10 real failures recovered) |
| Baseline-negative FPR (20 clean single-speaker negatives) | 15% | **20%** (1 new false positive) |
| Degraded-negative FPR (20 noisy/muffled negatives) | 10% | 10% (unchanged) |
| Added latency (CPU, per fallback attempt) | — | ~4-8s for a ~2.5s segment |

Confirmed via a full `evaluate_pipeline_corpus.py` run (not just the
isolated 10-failure/20-negative reproduction above), run back-to-back
against a no-fallback control on the same corpus: the one new baseline
false positive is `"I have a doctor's appointment tomorrow"`, where the
primary STT pass correctly returned nothing (`NO_ACTION`) but splitting
that clean clip into two estimated "speaker" streams hallucinated a fuzzy
match on one of them — exactly the failure mode predicted above, not a
new one. Degraded FPR's specific misfired transcript changed between runs
but the count didn't (that's `faster-whisper`'s own CPU run-to-run
nondeterminism, confirmed by the same shift appearing in the *control*
run's `muffled` recall split too, with zero fallback code involved).
**Bonus, not designed for:** the same full run also measured
`pitch_tempo_distorted` recall jumping from 45% to 65% from the
separation fallback alone, with `UDK_ENABLE_TIERED_STT` not even
enabled — splitting a distorted single-speaker clip into two streams
apparently cleans up the signal somewhat too, not just genuine overlap.

Real cost, not a free win: separation occasionally hallucinates a second
"speaker" out of clean single-speaker audio whose garbled transcript
clears `TRIGGER_VERIFY`, and the model itself is slow on CPU. That cost is
why this is wired as a **fallback that only ever runs after the primary
STT(+KWS) pass already returned `NO_ACTION`** — never on every segment,
unlike KWS/semantic which run alongside every segment. Opt-in via
`UDK_ENABLE_SEPARATION=1` (`api.py`'s `_load_separator`,
`evaluate_pipeline_corpus.py`'s `separator=` param), same pattern as
`UDK_ENABLE_KWS`/`UDK_ENABLE_SEMANTIC`. `retry_with_separation()` is safe
to call after a `NO_ACTION` primary decision because `UDKEngine.decide()`
never mutates repetition/distinct-UDK state on `NO_ACTION` — verified in
`test_separation.py`.

**Pitch/tempo distortion: a confidence-gated two-tier STT fallback
(`stt.py`'s `TieredWhisperSTT`).** `faster-whisper` already computes
`avg_logprob` per segment; `tiny.en`'s fast pass just never surfaced it.
Real measured margin: clean baseline audio never drops below avg_logprob
-0.458 (mean -0.385, n=20); the 12 real `pitch_tempo_distorted` failures
ranged -0.51 to -0.91 — no overlap, so -0.5 is a clean gate. Below that
threshold, the segment is re-transcribed with `base.en` and that result is
used instead. Real result:

| Metric | Before (tiny.en only) | After (tiered) |
| --- | --- | --- |
| `pitch_tempo_distorted` recall | 40% (12/20 failed, reproduced directly) | **75%** (7/12 real failures recovered) |
| New false positives (40 negative clips, clean+noisy/muffled) | — | **0** |
| Fallback trigger rate on those same negatives | — | 15/40 (noise alone also depresses `tiny.en`'s confidence) |

This one has a strictly better risk profile than the separation fallback
— a real ~15-point-plus recall gain with zero measured new false
positives, at the cost of `base.en` latency on roughly a third of noisy
segments (not just failing ones). Opt-in via `UDK_ENABLE_TIERED_STT=1` in
`evaluate_pipeline_corpus.py`; not wired into `api.py` because nothing
there currently env-selects the STT backend at all (`_stt_backend` is
swapped wholesale by whoever deploys it, same as it already was for
`FasterWhisperSTT`) — `TieredWhisperSTT` is a drop-in `STTBackend`
implementation, no other code needed to change to use it.

**What neither of these claims:** overlapping speech is still wrong 35%
of the time, and a generic 2-speaker separator trained on read speech
(WSJ0-2mix) has no exposure to this product's real audio conditions.
Distortion-robustness is better but still not 100%, and — per the TESS
finding — the underlying condition it's testing against was already
shown to probably overstate how much real distress delivery would cost
STT in the first place. Both are real, measured improvements on the
numbers as measured, not claims that either gap is closed.

## Stress-testing against the actual threat model: memorized-phrase recall drift

A real product question surfaced a methodology mistake before it became a
bad recommendation: "how do I check no real case is ever missed?" was
first answered by testing 18 open-ended distress sentences a random
stranger might say (`"He has a knife"`, `"I think he's going to kill
me"`) — only 61-67% were caught, and lowering `TRIGGER_VERIFY_THRESHOLD`
to fix it would have cost a measured 55% false-positive rate on ordinary
speech. **That test was measuring the wrong thing.** This product's
actual design (Section 5) shows the 20 UDKs to the user at login and asks
them to remember them — the real risk isn't a stranger's unrelated
distress speech, it's *this specific user's degraded recall of a phrase
they were shown*, a bounded problem unlike the open-ended one.

Re-tested correctly: 40 realistic "almost remembered it" variants (2 per
UDK — dropped words, reordering, synonym swaps of the exact memorized
phrase, not unrelated content). On clean audio: **100% fired some alert,
90% matched the exact intended UDK** — no threshold change needed at
all. Under the two hardest real conditions, though, near-miss recall
collapses just as badly as exact-phrase recall does — because the
failure mode is STT producing garbage (`"Please call the police"` →
`"at least all of you is."`), not a matching-tolerance problem:

| Condition | Any alert (no fallbacks) | With both fallbacks |
| --- | --- | --- |
| Baseline | 100% | 100% |
| Noisy | 100% | 100% |
| Muffled | 95% | 100% |
| `pitch_tempo_distorted` | 50% | **88%** |
| Overlapping | 52% | **78%** |

Enabling `UDK_ENABLE_SEPARATION`+`UDK_ENABLE_TIERED_STT` together recovers
most of that gap (recall above), but re-checking the false-positive side
(20 ordinary negatives × all 5 conditions, no-fallback control run back
to back) found a consistent +5 to +10 point FPR cost on *every* condition,
not just the ones being fixed:

| Condition | FPR, no fallbacks | FPR, both fallbacks |
| --- | --- | --- |
| Baseline | 15% | 20% |
| `pitch_tempo_distorted` | 15% | 25% |
| Muffled | 15% | 25% |
| Noisy | 15% | 20% |
| Overlapping | 15% | 20% |

**A real bug was found and fixed while checking this, not just a cost
measured.** Two of the new false positives had been silently upgraded
from `TRIGGER_VERIFY` to an unconfirmed `TRIGGER_ALL`. Diagnosing both
individually found two *different* causes, not one:
- `"I need to buy some groceries"` (overlapping) — a real bug in
  `separation.py`'s `retry_with_separation()`. It called `engine.decide()`
  directly on the live engine once per separated stream; two garbled
  readings of the *same single moment* each independently cleared 0.60 on
  different UDKs, and the distinct-UDK-escalation rule (built earlier for
  two different REAL phrases said in a row) mistook two simultaneous
  misreadings of one segment for that, forcing `TRIGGER_ALL` at only
  0.648 confidence. **Fixed**: each stream is now probed against a
  throwaway engine so simultaneous guesses can never cross-contaminate
  each other's window state; only the single winning candidate is applied
  to the real engine, exactly once. Verified against the real audio that
  originally exposed it (now correctly lands at `TRIGGER_VERIFY`, 0.648,
  `verify_by_default`) and with a new fast regression check
  (`test_separation.py`, 6/6 passing, up from 4/4).
- `"I need to go now"` (noisy) — **not caused by anything built this
  session.** The transcript was byte-identical to the no-fallback control
  run; what changed was the KWS layer's distance landing at 0.167-0.172
  (just under the 0.18 calibrated threshold) on one run and not the
  other — real `wav2vec2`/`faster-whisper` CPU inference nondeterminism
  (the same phenomenon behind the earlier `muffled` 19/1-vs-18/2 split)
  tripping the pre-existing STT+KWS corroboration-escalation rule
  (Section 4's Approach E, built long before separation/tiered-STT
  existed). Capping fallback-recovered confidence — the fix originally
  proposed — would not have touched this at all, since it never goes
  through the fallback path; corroboration forces `TRIGGER_ALL`
  regardless of raw confidence.

**(2) has since been fixed too, with real calibration data, not a
guess.** `kws.py`'s `spot()` already collapses its DTW distance into a
binary match/no-match at `DEFAULT_MATCH_THRESHOLD=0.18`, throwing away
*how* confidently it matched — a distance of 0.167 counted exactly the
same as 0.02 for corroboration purposes. Re-running
`calibrate_kws_threshold.py` for the raw distance distribution (not just
recall/FPR at one cutoff) found real separation: at 0.15, recall on that
corpus drops from 93.9% to 81.8%, but false-positive rate is **cut in
half** (10.0% → 5.0%). That asymmetry matters specifically for
corroboration: missing the boost only means the detection stays at the
still-safe, still-alerting `TRIGGER_VERIFY` tier (STT's own match stands
on its own); a *wrong* boost skips confirmation entirely. Added
`KWS_CORROBORATION_MAX_DISTANCE=0.15` (`udk_engine.py`) — stricter than
the general 0.18 used for KWS-alone detection (which stays permissive,
since that's the STT-outage safety net and shouldn't change), used only
to gate whether a KWS match is trusted enough to justify corroboration.
`_fuse()` now takes the raw `KWSMatch` (not just its converted
confidence) to check this. Verified three ways: the existing
corroboration test (`test_kws.py`, distance 0.15, right at the new
boundary) still passes; a new regression case at distance 0.17 (clears
0.18, not 0.15) now correctly stays `TRIGGER_VERIFY`; and both real
audio cases that originally exposed the bug were re-run and now
correctly land at `TRIGGER_VERIFY` instead of an unconfirmed
`TRIGGER_ALL`. Full regression suite: 155/155 checks passing (counted
directly via each file's `[PASS]` lines, not the per-file self-reported
totals -- a discrepancy against earlier session totals claimed elsewhere
in this doc, corrected here rather than propagated).

## Multilingual support: English, Hindi, Telugu, Kannada, Tamil

Added on explicit request ("i want it to be available for multilingual
languages, atleast in india for, english, hindi, telugu, kannada, tamil
atleast these 5"). KWS was scoped out deliberately (see below); STT,
UDK matching, and semantic paraphrase-catching now support all five.
Two real, previously-invisible bugs were found and fixed while building
this -- not just new phrase lists bolted on:

**Bug 1 -- `_normalize()` silently corrupted every non-Latin-script
phrase.** Its regex relied on Python's `\w`, which does not include
Unicode COMBINING MARKS (category Mn/Mc) -- the vowel signs (matras)
that carry essential meaning in Devanagari/Telugu/Kannada/Tamil. It
turned "मुझे" (Hindi "mujhe", = "me") into "मझ" (garbled, a different
string) by stripping its two vowel signs. **Fixed** by stripping actual
Unicode punctuation/symbol categories (P\*/S\*) instead of relying on
`\w`'s incomplete letter-vs-mark distinction -- verified to strip English
punctuation identically to the old behavior (`test_multilingual.py`)
while preserving Indic combining marks.

**Bug 2 -- the semantic model's "multilingual" claim, made and cited
throughout this project since it was picked, was never actually verified
against Telugu, Kannada, or Tamil.** `paraphrase-multilingual-MiniLM-L12-v2`
works well for Hindi (comparable separation to English) but is
near-useless for the three Dravidian languages at 0.55: real testing
found completely UNRELATED sentences ("I am hungry" vs. "someone is
trying to hurt me") scoring 0.56-0.89, all clearing the threshold --
not a calibration problem, the embeddings weren't discriminating meaning
at all in those languages (almost certainly because they're far lower-
resource than Hindi/English in its training data). **Fixed** by adding
`sentence-transformers/LaBSE` (trained explicitly for broad cross-lingual
retrieval across 109 languages) as a second matcher for the other four
languages, with its own separately-calibrated threshold -- a real
10-positive/200-negative-pair-per-language corpus (10 UDK paraphrases x
the existing 20 ordinary-negative sentences, translated) found clean,
consistent separation at **0.80** across all four languages:

| Language | Recall @ 0.80 | FPR @ 0.80 |
| --- | --- | --- |
| Hindi | 100% | 0.5% |
| Telugu | 100% | 0.5% |
| Kannada | 100% | 0.5% |
| Tamil | 100% | 0.5% |

LaBSE's raw scores run systematically higher even for genuine matches
(0.86-0.99 vs. the English model's 0.59-0.76), which is exactly why this
needed its own threshold, not a reuse of 0.55. `semantic.py`'s
`SemanticBackend` now carries its own `threshold`, and `udk_engine.py`'s
`match_transcript()` reads it from whichever matcher is given, defaulting
to 0.55 for anything that doesn't declare one (test doubles). English
keeps its original model + 0.55, untouched, no reason to risk regressing
already-validated behavior.

**Real, evidence-based translations, not guesses left unverified:** 20
UDKs translated into each language (`udks.py`'s `_UDK_TABLE`), with
`verify_by_default` flags carried over 1:1 from the English assignment
as a starting assumption -- explicitly NOT independently re-measured
against a real per-language false-positive corpus the way the English
list's two flag revisions were (see above). These translations were
NOT sourced from a professional translator or native-speaker safety
reviewer -- Section 16's "the safety team owns the final list" applies
at least as much here as to English, and that review still hasn't
happened. What HAS been done, honestly stated as a substitute check, not
a replacement for one: **back-translation QA** (`facebook/nllb-200-distilled-600M`,
a real machine translation model) translated all 80 phrases back to
English and compared meaning against the original via the English
semantic matcher. 4/80 flagged (round-trip similarity < 0.70), all
investigated individually rather than accepted or dismissed on the
score alone:
- Hindi/Kannada UDK_09 ("I'm scared, stay back"): back-translated as
  "I'm afraid, stay away from me" / "I'm scared, away from you" -- a
  real but harmless paraphrase, not an error.
- Kannada/Tamil UDK_12 ("Someone is trying to hurt me"): both
  back-translated to nonsense ("I'm not sure what I'm doing"). Verified
  this is NLLB's OWN weakness in the Kannada→English direction, not a
  translation error: NLLB's own forward-translation of the English
  original into Kannada produces DIFFERENT Kannada text that back-
  translates to the SAME nonsense -- the failure is direction-specific
  to the QA tool, not present in the original text. Confirmed further by
  the strongest evidence available: both phrases already passed their
  real end-to-end audio test (Kannada: exact match, confidence 1.00;
  Tamil: semantic match, confidence 0.88, both `TRIGGER_ALL`) run
  earlier this session -- a real TTS voice speaking the phrase and a
  real fine-tuned STT model recognizing it correctly is stronger
  evidence of correctness than any generic back-translation check.

**What this does and doesn't establish:** back-translation QA is a real,
verifiable check that catches outright mistranslation -- it caught
nothing that turned out to be a real error here. It CANNOT judge natural
phrasing, register, or whether a phrase is one a real person would
actually say under distress in that language, the same way the English
list's own two revisions (found by real false-positive testing, not
translation review) required native judgment neither this check nor
back-translation can substitute for. The safety-team review gap is
narrowed, not closed.

**Real end-to-end audio validation, all four languages, with genuinely
different results per language -- not one shared finding.** Real
MMS-TTS-synthesized audio (`facebook/mms-tts-{hin,tel,kan,tam}`) through
the real `VAD -> STT -> UDKEngine` pipeline, all 20 UDKs per language:

| Language | "small" model | "medium" model | Other attempts |
| --- | --- | --- | --- |
| Hindi | **90%** (18/20) | -- | -- |
| Tamil | **90%** (18/20) | -- | -- |
| Kannada | 0% (0/20) | **45%** (9/20) | -- |
| Telugu | 0% (0/20) | 5% (1/20) | `initial_prompt` script bias: 0/20, worse (hallucination) |

Hindi and Tamil's story: the "tiny" multilingual Whisper model wasn't
reliably transcribing into native script at all -- some output was
romanized ("Koi mera pichak kar raha hai" instead of "कोई मेरा पीछा कर
रहा है"), some garbled into entirely different scripts (Japanese/
Korean-looking characters), meaning it sometimes misidentified the
spoken language outright. Switching to "small" fixed script fidelity
completely and recovered 18/20 for both. The 2 remaining misses per
language are genuine STT transcription errors on short/fast phrases, not
a matching-layer problem. **`tiny`/`tiny.en` is not viable for
non-English deployment; `small` works well for Hindi and Tamil.**

Kannada and Telugu needed real further investigation, not just a bigger
model, because their failures looked different from each other:
- **Kannada** stayed in the correct script throughout but the phonetic
  transcription itself was too garbled to match (repeated/dropped
  syllables) -- a genuine STT-capacity bottleneck. "medium" model
  (larger than "small") recovered 9/20, a real improvement but not a
  fix; a still-larger model is the natural next step, not yet tried
  (real memory/time cost -- one background test was killed by a genuine
  system low-memory event mid-session).
- **Telugu** kept transcribing into DEVANAGARI script (Hindi's script)
  even at "medium," phonetically close but on the wrong characters
  entirely -- not a capacity problem, since more capacity didn't fix
  it (0% -> 5%, barely moved). The standard mitigation for this exact
  failure mode -- seeding `initial_prompt` with real Telugu-script text
  to bias the decoder -- was tried and made things WORSE, not better:
  it did force Telugu script, but the transcription collapsed into
  repetitive hallucinated nonsense (a documented Whisper failure mode
  when a prompt over-anchors a low-resource language's decoder).
  **Telugu real-time-speech detection does not currently work with any
  `faster-whisper` configuration tried** -- this is stated as an open,
  unsolved problem, not glossed over.

Both Hindi and Tamil's results confirm the whole architecture (UDK
translation, `_normalize()` fix, LaBSE matching, API wiring) works
correctly end to end on real audio, not just at the text level. Kannada
and Telugu initially confirmed the same architecture is only as good as
the STT model underneath it, and that model's real-world quality varies
sharply by language in ways that don't reduce to "bigger model always
helps."

**All four languages were then actually solved, not left as open
problems, once the right pretrained model was used instead of a bigger
generic one** -- the exact "compare pretrained models, don't just scale
the wrong one" discipline this project already used for KWS (wav2vec2
vs. two rejected CLAP formulations) and semantic matching (LaBSE vs. the
Dravidian-language-blind MiniLM model). `vasista22`'s community Whisper
fine-tunes, each trained on real speech in ONE specific Indic language
(not a generic multilingual model spread thin across 99 languages), were
found on HuggingFace and tested the same way, for all four:

| Language | Generic multilingual (best) | Language-specific fine-tune |
| --- | --- | --- |
| Telugu | 5% (`medium`, wrong script) | **100%** (20/20, `vasista22/whisper-telugu-medium`) |
| Tamil | 90% (`small`) | **100%** (20/20, `vasista22/whisper-tamil-medium`) |
| Kannada | 45% (`medium`, garbled) | **95%** (19/20, `vasista22/whisper-kannada-medium`) |
| Hindi | 90% (`small`) | **95%** (19/20, `vasista22/whisper-hindi-medium`) |

The fine-tune won for every language, not just the two that were
completely broken with the generic model -- confirming this is a real,
general improvement, not a fix limited to the languages that were
failing outright.

**Critical real bug found and fixed after the accuracy numbers above
were already measured: real-time latency was never checked until asked,
and the honest first answer was "this is nowhere close to viable."**
These fine-tunes only ship as `transformers`-format checkpoints
(`pytorch_model.bin`), not `faster-whisper`'s CTranslate2 format, so the
first working version of `stt.py`'s `IndicFineTunedWhisperSTT` ran them
via plain `WhisperForConditionalGeneration.generate()` on CPU. Measured
real per-segment latency: **37.29 seconds for a single 2-second clip**
(Telugu, "medium") -- for a product whose entire premise is detecting
distress and alerting within seconds, this was a genuine, serious defect
hiding behind otherwise-good accuracy numbers, not a minor rough edge.

Root-caused and fixed properly, not patched around:
1. **Wrong runtime.** Plain PyTorch autoregressive `generate()` on CPU is
   dramatically slower than CTranslate2's optimized inference -- the same
   gap that makes English's `tiny.en` viable for real-time in the first
   place. `ctranslate2` (already a transitive dependency of
   `faster-whisper`) can convert any HuggingFace Whisper checkpoint,
   including these fine-tunes, into its own format via
   `ctranslate2.converters.TransformersConverter` -- a real, one-time
   ~1-2 minute conversion, not a new dependency. Converting alone cut
   Telugu "medium" from 37.29s to 14.41s (2.6x) with **zero accuracy
   change** (still 100%).
2. **Wrong model size, never actually checked.** "medium" was the first
   size tried, not a measured choice -- the same mistake the original
   0.30 KWS threshold and 0.55-everywhere semantic threshold were, both
   already caught and fixed earlier by measuring instead of assuming.
   Every available size was converted and measured for real, for every
   language:

| Language | Size | Recall | Mean latency |
| --- | --- | --- | --- |
| Telugu | medium | 100% | 14.41s (37.29s uncoveted) |
| Telugu | small | 100% | 5.97s |
| Telugu | **base** | **100%** | **1.90s** |
| Kannada | small | 90% | 6.99s |
| Kannada | **base** | **95%** | **2.33s** |
| Hindi | **small** (no base exists) | **95%** | **5.99s** |
| Tamil | **small** (no base exists) | **100%** | **6.81s** |

**Telugu and Tamil had zero accuracy cost at their fastest available
size; Kannada's "base" actually beat "small" on both axes at once**
(95% vs. 90%, 2.33s vs. 6.99s) -- there was no real speed/accuracy
tradeoff to make for three of the four languages, and picking "medium"
originally was simply never justified by evidence. `INDIC_FINETUNED_MODELS`
now points at the real winning size per language (`telugu-base`,
`kannada-base`, `hindi-small`, `tamil-small`), and `IndicFineTunedWhisperSTT`
converts + caches the CTranslate2 model on first use
(`indic_ct2_models/`, alongside this checkout, same pattern as
`kws_references.npz`) and runs it through `faster_whisper.WhisperModel`
from then on -- the exact same fast runtime English already uses, now
shared by all five languages.

A second real bug, found and fixed while wiring the original version of
this class: this checkpoint family's tokenizer doesn't register its
special tokens (`<|startoftranscript|><|te|><|transcribe|>
<|notimestamps|>`) in a way `skip_special_tokens=True` catches -- every
raw transcript came back with that literal text prefixed. It didn't
break the accuracy numbers (the exact-match layer's substring check
happened to tolerate the extra prefix), but relying on that happenstance
would be fragile -- a real deployment could hit a shorter UDK phrase
colliding with stray tokenizer text by bad luck. Fixed by explicitly
stripping `<|...|>`-pattern text from the decoded output, kept in the
rewritten class as a defensive measure even under the new runtime.

**What this means for "is this actually real-time":** Telugu/Kannada now
sit at ~2 seconds per segment -- comparable to English's own SLO target
(p50 ≤ 1.5s for the *whole* pipeline, not just STT) and genuinely usable.
Hindi/Tamil sit at ~4-6 seconds after a further free optimization (below)
-- usable, but not yet at English's level, because neither has a smaller
fine-tuned variant available from this model family to test. This is
stated as the honest remaining gap, not smoothed over: three of five
languages are demonstrably real-time-fast, two are "usable but slower,"
and closing that last gap needs either a smaller Hindi/Tamil fine-tune
(none found yet) or accepting the current figure.

**A second free latency win, also measured not assumed:**
`faster-whisper`'s default `beam_size` is 5; dropping to `beam_size=1`
(greedy decoding) was tested across all four languages and found ZERO
accuracy cost every time, for a real 7-26% latency reduction:

| Language | beam_size=5 | beam_size=1 | Recall (both) |
| --- | --- | --- | --- |
| Telugu | 2.53s | 1.88s | 100% |
| Kannada | 1.57s | 1.46s | 95% |
| Hindi | 3.91s | 3.60s | 95% |
| Tamil | 5.59s | 4.50s | 100% |

`IndicFineTunedWhisperSTT.transcribe()` now passes `beam_size=1` by
default.

**Degraded-condition and false-positive testing, completed for all four
languages** -- the same noisy/muffled/overlapping/pitch-tempo-distorted
corpus English was tested against (`audio_augment.py`, language-agnostic
signal processing), run for real against each language's fine-tuned STT
+ LaBSE matching:

| Condition | English | Telugu | Kannada | Hindi | Tamil |
| --- | --- | --- | --- | --- | --- |
| Baseline | 100% | 100% | 90% | 100% | 100% |
| Muffled | 100% | 100% | 85% | 95% | 100% |
| Noisy | 100% | 100% | 90% | 100% | 100% |
| Pitch/tempo distorted | 45% | 45% | 50% | 30% | 30% |
| Overlapping | 50% | 50% | 45% | 30% | 45% |
| FPR (baseline negatives) | 15% | 25% | 35% | 15% | 35% |
| FPR (degraded negatives) | 10% | 20% | 20% | 15% | 30% |

Real, honest, non-uniform findings, not a clean "multilingual works just
as well" story:
- **Telugu's hard-condition numbers land almost exactly on English's**
  (45%/45% distorted, 50%/50% overlapping) -- strong evidence the
  difficulty here is inherent to the audio-degradation technique itself
  (shared code, `audio_augment.py`), not language-specific.
- **Kannada is measurably the most fragile language** -- notably weaker
  than Telugu/English on muffled (85% vs. 100%) and noisy (90% vs. 100%),
  and the highest baseline FPR (35%). Consistent with Kannada being the
  hardest case throughout this whole exploration (needed the most model
  iterations to get right).
- **Hindi and Tamil are measurably worse on the two hardest conditions**
  (30% each on `pitch_tempo_distorted`, vs. Telugu/English's 45%) despite
  otherwise-solid baseline/muffled/noisy numbers -- a real, unexplained
  language-specific weak spot, not yet root-caused.
- **None of the four languages have the `UDK_ENABLE_SEPARATION`/
  `UDK_ENABLE_TIERED_STT` fallbacks available** -- those were built and
  calibrated against English only. Applying and re-calibrating them for
  these four languages (their real thresholds/behavior would need
  separate verification, not assumed to transfer) is a natural next step,
  not yet done.
- **FPR was higher than English's across the board** for Telugu/Kannada/
  Tamil (English 15%/10% vs. 25-35%/20-30%) -- **root-caused and fixed**
  (below): the fuzzy layer's raw 0.55 gate, tuned once for English, was
  never re-checked per language.

**FPR gap root-caused and fixed.** Real per-language diagnosis (translate
the same 20 ordinary negatives via NLLB, synthesize via MMS-TTS,
transcribe with each language's real fine-tuned STT, run through the real
`match_transcript`) found the false positives above were overwhelmingly
coming from the fuzzy layer specifically, not semantic -- e.g. Tamil: 5
of 6 false positives were fuzzy-layer, only 1 semantic. A follow-up sweep
(same real audio, raw `partial_ratio` scores cached, thresholds swept
cheaply against them) found WHY: false positives cluster at 0.65-0.81 raw
ratio and genuine UDK matches at 0.70-1.0 -- these overlap badly at the
existing 0.55 gate but cleanly separate by 0.80, and this is true for
**all four languages, including Hindi** (not just the three Dravidian
ones -- Hindi measured 60% FPR at 0.55 on this real audio too, just
masked in the table above by a different negative corpus):

| Language | FPR @ 0.55 (old gate) | FPR @ 0.80 (new gate) | Fuzzy-layer recall cost |
| --- | --- | --- | --- |
| Hindi | 60% | **0%** | none (20/20) |
| Telugu | 60% | **0%** | none (20/20) |
| Kannada | 40% | **0%** | 1/20 (still likely caught by LaBSE's own 0.80 gate -- same phrase, minor ASR noise, not a real paraphrase) |
| Tamil | 65% | **5%** | 1/20 (same reasoning) |

Fixed by giving the fuzzy layer its own per-matcher `fuzzy_threshold`
(`semantic.py`), same pattern already used for the semantic layer's
`threshold`: English keeps 0.55 (untouched, already validated), the
shared multilingual matcher now uses `MULTILINGUAL_FUZZY_THRESHOLD = 0.80`.
`udk_engine.py`'s `match_transcript()` reads it via
`getattr(semantic_matcher, "fuzzy_threshold", 0.55)`, same fallback
pattern as the semantic threshold, so callers with no opinion (test
doubles) keep the original 0.55 behavior. 3 new regression checks in
`test_semantic.py` (8 total) cover the default gate, a matcher declaring
a stricter gate rejecting a ratio that would've cleared 0.55, and the
backward-compat fallback.

**Investigating the pitch/tempo-distorted/overlapping weak spot found a
second, more serious bug: generic short fragments could fire an
UNCONFIRMED full alarm on the wrong UDK.** Real per-clip transcript
inspection (not just the aggregate recall number, which the
degraded-condition table above already flagged as "unexplained") under
`pitch_tempo_distorted`/`overlapping` found Hindi's and Kannada's
fine-tuned STT occasionally collapsing to a single bare, generic word
under heavy distortion instead of a garbled sentence -- e.g. Hindi
"मुझे" ("me"). `udk_engine.py`'s exact-match layer's substring check has
no length/uniqueness guard, so this fragment scored confidence=1.0 (an
UNCONFIRMED `TRIGGER_ALL`, bypassing `verify_by_default` entirely) on
whichever UDK happened to be first in iteration order to contain it --
and it usually wasn't the right one. Checking the corpus confirmed this
isn't Hindi/Kannada-specific: it's a structural bug present in **every**
language's UDK list, including English ("me" is a literal substring of
10 of English's own 20 UDK phrases, "help" of 5) -- it simply hadn't
surfaced yet because English's STT hasn't been observed collapsing this
far under distortion. The fuzzy layer independently has the exact same
vulnerability (rapidfuzz's `partial_ratio` scores a literal substring
alignment as ~100% regardless of length), so discounting only the exact
layer wouldn't have closed it.

Fixed with `udk_engine.py`'s new `GENERIC_FRAGMENT_CONFIDENCE` (0.65,
reusing the existing `KWS_MATCH_CONFIDENCE` value for consistency): a
transcript that's a substring of **more than one** UDK's phrase is
capped at that confidence in both the exact and fuzzy layers -- still
registers as something worth a confirmation prompt (this project's
established bias toward false positives over silent misses), but can no
longer skip straight to an unconfirmed full alarm; a fragment that's a
substring of only one UDK keeps full confidence, since corpus-relative
uniqueness is real, unambiguous evidence regardless of how short it is.
Verified via the same real transcripts that found the bug: every
capped case landed on 0.650/`TRIGGER_VERIFY` instead of 1.000/unconfirmed
`TRIGGER_ALL` on a wrong UDK, and all 7 fragments across the same test
run that legitimately remained at full confidence were independently
confirmed correct (`correct=True`) -- zero remaining false-full-confidence
cases in that run. Re-ran English's full corpus evaluation afterward:
recall (100/100/100/45/50%) and FPR (15%/10%) both came back byte-for-byte
identical to before, confirming no regression (English's own STT doesn't
happen to produce these bare fragments in that test corpus). 2 new
regression checks added to `test_semantic.py` (10 total) using a small
synthetic UDK pair to verify both the ambiguous-fragment cap and the
unique-fragment pass-through deterministically, without depending on
real STT ever reproducing the failure mode. Full regression suite:
**183/183 checks passing.**

**Checked whether `kws.py`'s DTW-based audio matching shares this same
bug -- real test found it does NOT.** Synthesized the exact fragment that
caused the text-layer bug ("मुझे") plus another generic word ("मैं") and
spotted them against the real English and Hindi reference banks
(reporting every phrase's DTW distance, not just the winner, so ambiguity
would actually be visible). Both came back roughly 2-2.7x the match
threshold (Hindi: 0.342 and 0.541 vs. a 0.20 threshold) -- cleanly
rejected, not a false match, and genuine full phrases still matched
cleanly with a huge margin to second place (Hindi: 0.056 vs. 0.245).
Root cause of the difference: `_dtw_distance`'s cost is normalized over
the FULL alignment path length, so a short query aligned against a much
longer reference accumulates real cost for the reference's unmatched
content -- unlike a text substring check or `partial_ratio`, which treats
"found it somewhere" as a free, length-blind perfect match. The
vulnerability was specific to the text layers' scoring mechanics, not
something DTW-over-frame-embeddings inherits. See `kws.py`'s module
docstring for the same finding in context.

Note on the degraded-condition recall percentages themselves (the table
above, and this section's own before/after numbers): re-measuring found
them noticeably noisy run-to-run (e.g. Telugu's `pitch_tempo_distorted`
recall read 45% originally, 35% after the FPR fix, 15% after this fix --
on the same 20 UDKs, same augmentation code) even when the only
relevant change between two runs was a code path this specific
language's STT rarely exercises. Root cause: MMS-TTS (`VitsModel`) uses
a stochastic duration predictor/flow, so re-synthesizing "the same"
phrase produces audibly different audio each run, which real
distortion+a real STT model can transcribe differently -- most of the
run-to-run swing is TTS synthesis randomness, not a code regression.
This means the per-run recall percentage alone is not a reliable
before/after signal for a code change this size; the per-clip transcript
inspection above (a fixed, inspectable failure mode, confirmed present
and confirmed fixed) is the credible evidence here, not the aggregate
number. The original unexplained-weakness question (why Hindi/Tamil
specifically read weaker than Telugu on paper) is now partially
answered: at least part of the apparent language-specific gap was this
matching-layer bug interacting with each language's STT failure mode
differently (Hindi/Kannada's STT collapses to bare words; Telugu/Tamil's
tends toward full garbled sentences instead) rather than a real
difference in underlying audio robustness -- the remaining gap, if any,
would need a much larger/fixed (not freshly re-synthesized) audio corpus
to measure past this TTS-randomness noise floor, not yet done.

**Wired into the live API, not just proven in isolated test scripts.**
`api.py`'s `create_or_get()` takes `stt_by_language`, a
`{language: STTBackend}` override map tried before falling back to the
default shared STT backend -- same pattern as the multilingual semantic
matcher. `_load_indic_stt_backends()` loads all four real fine-tunes
behind their own opt-in env var (`UDK_ENABLE_INDIC_STT=1`, separate from
`UDK_ENABLE_SEMANTIC` since this is its own real download+convert cost
on first run -- a few hundred MB per language now that the winning
sizes are `base`/`small` rather than `medium`, not the ~1.5GB per
language the original unmeasured "medium" choice would have cost). Any
of the four non-English journeys started through the real API now
actually uses its fine-tuned model, converted to CTranslate2 and cached
in `indic_ct2_models/`, not just in a standalone measurement script.
`test_multilingual.py` (21 checks) adds fast coverage of the selection
logic itself via fake STT backends, with no real model load needed for
the routine suite.

**KWS was initially scoped out, then built once asked to.** Its
wav2vec2 model (`facebook/wav2vec2-base-960h`) is English-only; making
it multilingual meant a real model search, not a quick swap.

**First attempt failed cleanly and informatively.** `facebook/wav2vec2-large-xlsr-53`
(a raw self-supervised multilingual model, the obvious first choice) gave
**zero usable signal**: 0/10 real Telugu paraphrases matched their own
enrolled phrase as closest, and positive/negative distances completely
overlapped (both in the 0.000-0.025 range). Also crashed outright on
load -- `Wav2Vec2Processor.from_pretrained()` needs a CTC vocabulary
this raw checkpoint never had, since it was never fine-tuned for ASR
(fixed by switching `kws.py` to `Wav2Vec2FeatureExtractor`, which is all
this DTW approach actually needs -- it never decodes text). Root cause
of the zero-signal result: this DTW approach needs a model fine-tuned
for ASR (which sharpens word/phrase-level structure in its hidden
states), not just self-supervised pretrained -- confirmed by the fact
English's own KWS model already IS ASR-fine-tuned (960 hours of
transcribed English), which turns out to be *why* it works, not
incidental.

**Fixed by switching to the Vakyansh project's per-language ASR-fine-tuned
wav2vec2 models** -- the same "find the model actually trained for this,
don't just scale a generic one" pattern as the STT fix. Real per-language
calibration (10 paraphrase positives + 20 negatives each -- lighter than
English's 120-clip corpus, a real starting point not a finished
calibration):

| Language | Model | Correct-closest | Threshold | Recall | FPR |
| --- | --- | --- | --- | --- | --- |
| Telugu | `vakyansh-wav2vec2-telugu-tem-100` | 8/10 | 0.25 | 80% | 15% |
| Kannada | `vakyansh-wav2vec2-kannada-knm-560` | 7/10 | 0.20 | 60% | 30% |
| Hindi | `vakyansh-wav2vec2-hindi-him-4200` | 8/10 | 0.20 | 90% | 10% |
| Tamil | `vakyansh-wav2vec2-tamil-tam-250` | 7/10 | 0.25 | 70% | 25% |

Hindi's KWS quality is genuinely close to English's own calibrated
numbers; Kannada is the weakest, consistent with it being the hardest
language throughout this whole exploration. `enroll_indic_kws_references.py`
(mirrors `enroll_kws_references.py`, using MMS-TTS instead of Windows
SAPI voices) generates `kws_references_{te,kn,hi,ta}.npz`. Wired into
`api.py` via `_load_indic_kws_backends()` and `create_or_get()`'s new
`kws_by_language` param (same override-map pattern as `stt_by_language`),
gated behind its own `UDK_ENABLE_INDIC_KWS=1` (separate from
`UDK_ENABLE_INDIC_STT` since this is its own real model set). Verified
end to end through the live API: a real Telugu journey's KWS backend
correctly spots real synthesized UDK_05 audio (distance 0.028, well
under threshold).

**API wiring**: `StartRequest` takes a `language` field
(`udks.py`'s `SUPPORTED_LANGUAGES`). **Update: the client no longer
declares this at all -- see "Automatic language detection" below, which
replaced the client-declared field with real detection from the
journey's own audio.** The journey's language is still persisted
(`db.py`/`db_postgres.py`, same lazy-migration pattern as `had_trigger`)
so crash recovery resumes the last-detected language rather than
restarting detection from scratch. `test_multilingual.py` (28 checks)
covers the `_normalize()` fix, exact-match correctness for every
language, the automatic-detection and periodic-recheck logic (below),
and both the `stt_by_language` and `kws_by_language` selection logic.

## Automatic language detection: no client input, detect from real audio

**The problem this closes**: the client (whatever app calls this API)
was trusted to declare the speaker's language up front, once, for the
whole journey. A real deployment can't always guarantee that -- the
client may not know, may guess wrong, or (in this project's case) simply
won't ever send it. Asked directly: "the audio can be Telugu, Hindi,
English, whatever" -- the system needs to figure this out from the audio
itself, not be told.

**Two real, complementary models, chosen from real measurement, not
assumption** (`language_id.py`):

| Detector | Real accuracy (this project's own test) | Real weakness |
| --- | --- | --- |
| `faster-whisper`'s built-in language detection (a single encoder pass, no decoding -- fast) | 100% on English/Hindi/Telugu/Tamil, real short (2-4s) clips | 67% on Kannada -- confused with linguistically close languages (Sinhala, Tamil) |
| SpeechBrain's VoxLingua107 (ECAPA) -- purpose-built for language ID, not a byproduct of an ASR decoder (the same "purpose-built beats generic" pattern already found for STT/semantic matching/KWS in this project) | **100%** on Hindi/Telugu/Kannada/Tamil -- fixes the Kannada weakness completely | Needs **6+ seconds** of audio to be reliable -- shorter clips can be badly wrong (real measured: Maltese/Croatian/Latin guesses on 1.8-4s English clips) |

**Design, using each one's real strength**: the fast Whisper detector
picks an initial language as soon as *any* audio exists, so a journey
never sits undetected waiting for enough audio to accumulate -- alarms
still need to fire within seconds. VoxLingua107 then rechecks
periodically once 6+ seconds have accumulated since the last check, and
**two consecutive agreeing rechecks are required before actually
switching** a journey's language mid-stream -- one disagreement can be
noise, two in a row is a real signal (the same reasoning already used
for `udk_engine.py`'s `REPEAT_WINDOW_S`). A confirmed switch rebuilds the
journey's STT/KWS/semantic-matcher/UDK-engine for the new language;
switching resets repetition/distinct-UDK tracking, which is correct --
a genuine language switch is a new context, not a continuation.

Gated behind `UDK_ENABLE_LANGUAGE_ID=1` (same opt-in-heavy-model pattern
as `UDK_ENABLE_INDIC_STT`/`UDK_ENABLE_SEMANTIC`); with it unset, a
journey simply stays on English defaults forever, the same honest
fallback this project uses everywhere else a heavy model is optional.

**Verified end to end through the live API, with real models, not just
fakes**: a journey created with `UDK_ENABLE_LANGUAGE_ID=1` and
`UDK_ENABLE_INDIC_STT=1`, no language field at all, fed 3.3 real seconds
of MMS-TTS-synthesized Telugu audio -- correctly detected `te`, routed to
the real `whisper-telugu-base` fine-tuned backend, and fired a real
`TRIGGER_VERIFY` on the correct UDK (UDK_08) despite the STT garbling
part of the transcript (`'దయచేసి గ్రా'` vs. the full phrase), exactly the
kind of case the fuzzy/semantic layers exist to catch. `test_multilingual.py`
covers the fast-detection, no-detector-fallback, and two-consecutive-recheck
hysteresis logic with fake detectors (no real model load in the routine
suite).

**Honest open item**: this replaces a client-declared language with two
detectors' real measured accuracy (93-100% depending on language and
audio duration) -- not 100% for every language at every point in a
journey. A wrong initial guess on a very short first utterance is
possible (matches Whisper LID's real 67% Kannada rate) and self-corrects
once the VoxLingua107 recheck has enough audio, typically within the
first 12-18 seconds of a real journey (two recheck cycles). This has not
yet been measured against degraded/noisy real audio the way STT/KWS
were -- only clean synthesized speech, same starting-point caveat as
several other calibrations in this project.

## Real-world audio testing: three real bugs found that no synthetic test surfaced

Real recordings (podcast clips, short-form videos, spanning English/
Hindi/Telugu/Kannada/Tamil, 8-60 seconds each) were dropped into
`real_recordings/{positives,negatives}/` and run through the actual
production journey pipeline -- streamed in chunks via `_ingest_frame`,
exactly like a live WebSocket journey, not a single-shot helper. This
found three real, previously-invisible bugs, all fixed:

**Bug 1: segments that never close on continuous real-world audio.**
`VAD.segment_speech`'s loop (`while j < n and flags[j]: j += 1`) only
stops at a non-speech frame. A real 56-second video clip had no
VAD-detectable pause anywhere in it (continuous talk/background sound) --
webrtcvad classified it as one unbroken speech run, so the segment kept
growing to match the buffer every time, and `api.py`'s finalization proof
(`seg.end_ms < total`, "more audio arrived after this ended") could never
become true. The entire clip stayed one open segment and was **never
transcribed or matched until the journey was explicitly stopped** --
silently falling back to "wait for everything," exactly the behavior
this system is supposed to never have. Fixed with a real
`max_segment_ms` cap (8000ms, a deliberate starting choice, not yet
swept against real data) in `vad.py`: a continuous run gets forcibly cut
at the cap and finalized, with pre-roll/post-roll overlap at each cut so
a phrase split across it isn't dropped from both sides. Verified
directly on the real clip that exposed it: 1 never-closing segment
became 6 real, incrementally-processed ones. `test_vad.py` (new, 7
checks) covers this directly -- `VAD.segment_speech` had no direct unit
test before this, only indirect coverage through fake test doubles
elsewhere.

**Bug 2: English never had a real STT backend wired into the live API at
all.** Every other backend here (KWS, semantic matching, Indic STT,
speech separation, language-ID) has a real opt-in loader gated behind an
env var. English's `_stt_backend` was hardcoded to `MockSTT()` (a test
double that always returns an empty transcript) with **no way to enable
a real one** -- found because a real English clip, correctly
language-detected, produced zero STT calls at all through the actual
`/start` -> journey pipeline. In production this meant an English
journey got zero real transcription-based detection, silently -- only
KWS (if separately enabled) could ever fire for English. Fixed with
`_load_stt_backend()` (same pattern as everything else), gated behind
`UDK_ENABLE_STT=1`, defaulting to `faster-whisper`'s `tiny.en` (matching
every other English test/eval script in this project).

**With both fixed, real detection on real audio, for the first time in
this project:**
- The real 56-second positive clip fired 3 real events across its (now
  correctly segmented) speech -- 1 `TRIGGER_VERIFY` and 2 `TRIGGER_ALL`,
  on transcript fragments genuinely resembling distress language ("let
  me go," "let's get out of here," "I lost him") from what appears to be
  a dramatized/acted scene. This is an encouraging real signal, stated
  carefully: it's acted content, not a scripted UDK-phrase reading, so
  it's not a clean ground-truth true-positive the way a real person
  saying an actual UDK phrase would be.
- One of the real negative clips produced a **genuine false positive**
  (`TRIGGER_VERIFY` on UDK_16, fuzzy layer, confidence exactly 0.60) on
  ordinary English content ("...he is presenting our film in Canada")
  -- this is the same fuzzy-layer short-phrase-inflation pattern already
  documented for English elsewhere in this project, now confirmed on
  real (not synthetic) speech for the first time, because fixing Bug 2
  is what let English STT run on this clip at all. It had been
  invisible until now, masked by the missing-STT bug.
- A real correction to an assumption made mid-investigation, worth
  stating plainly: a clip titled with a `#kannada` hashtag turned out to
  be genuinely English-language content (real Indian media frequently
  code-switches or is tagged by topic/actor language, not literal
  spoken-audio language) -- the language detector's "en" result for that
  clip was very likely correct, not a detection failure as first
  assumed. A different clip in the same batch (the same one that surfaced
  the false positive above) appeared, at first, to show real code-switching
  between English and Kannada mid-recording -- see Bug 3 below for why
  that read turned out to be wrong too.

**Bug 3 (found investigating what looked like real code-switching):
the periodic language recheck could confidently switch to the WRONG
language on real noisy audio.** `language_id.py`'s own real accuracy
measurement for VoxLingua107 (100% correct on all four Indic languages,
0.99+ confidence) was measured on clean synthetic TTS audio -- the same
kind of gap this project has hit three times before (see
"Generic multilingual models underperform" -- except this time the gap
isn't generic-vs-specialized, it's clean-vs-real-noisy audio for a
model that WAS the specialized, purpose-built choice). Direct
measurement against the real recordings (`diagnose_lid_flipflop.py`)
found VoxLingua107 landing on the wrong language with HIGH confidence
for two consecutive real 6-second windows in a row on real English
content -- `kn` at 0.88, then `kn` at 0.56, on a clip whose real
transcript is plainly English ("he is presenting our film in Canada").
Since `_maybe_update_language` only required 2 consecutive agreeing
rechecks before switching (same reasoning as `udk_engine.py`'s
`REPEAT_WINDOW_S`: one disagreement can be noise, two in a row was
assumed to be real), this real noise burst was enough to wrongly flip
the whole journey onto the Kannada STT/KWS backends mid-stream --
producing garbled, meaningless Kannada-script "transcripts"
(`'ಯಂತ್ರದ ಬೋಳಿ ಮಕ್ಕಳ ...'`) for real English speech, and losing all
detection capability for that stretch of the journey (no exact/fuzzy/
semantic layer can match English UDK phrases against Kannada-script
garbage). What looked like genuine code-switching in the earlier
write-up was this misdetection, not the audio actually changing
language. Fixed by raising the required consecutive agreements from 2
to 3 (`api.py`'s `LID_SWITCH_CONFIRMATIONS`), generalized from a single
pending-candidate comparison to a real streak counter. Verified two
ways: (1) replaying the real captured VoxLingua score sequences from
both real negative clips through the new logic confirms the wrong
switch no longer fires, while the (already-correct) "never switch, stay
English" outcome for the other clip is unaffected; (2) re-running the
real audio end-to-end confirms the journey now stays on English the
whole way through, with only its one initial resolution logged, no
mid-stream flip at all. `test_multilingual.py` gained a direct
regression test replaying this exact real score sequence (`kn`, `kn`,
`en`) so this specific real bug can't silently come back. Honest cost:
a genuine language change now takes ~18s of audio to confirm instead of
~12s -- accepted, since a wrong switch actively destroys detection
capability while a slightly slower correct one just delays it.

Also noticed while re-verifying, and explicitly NOT part of this fix:
`FasterWhisperSTT`'s transcripts for the same real clip weren't
byte-identical across repeated runs (e.g. "I'm" vs "I am", and
occasionally a materially different sentence for the same audio
segment) -- real inference nondeterminism on CPU, not something the
language-detection fix touches. Stated honestly as an open, separate
observation rather than folded into this fix's verification.

Full regression suite as of this writing: **197/197 checks passing**
(`test_pipeline.py` through `test_multilingual.py`, plus `test_vad.py`,
counted directly via each file's `[PASS]` lines).

**A second real positive clip (`_converted_positives_1.wav`, Tamil, 35s)
was added and run the same way afterward.** Its first ~18s is an
English narrator's boilerplate outro ("thank you for watching"); the
language detector correctly stayed on `en` through it, then switched to
`ta` after 3 consecutive agreeing rechecks (~18s in) once real Tamil
content started -- a live confirmation that Bug 3's fix (raising
`LID_SWITCH_CONFIRMATIONS` 2->3) still correctly detects a GENUINE
language change, not just suppresses false ones. The Tamil content
itself is narration about a film plot (actors Simran, Jothika, Suriya),
matching the clip's "kidnap" title -- descriptive narration, not a
distress utterance, and correctly produced no detection. One event did
fire: `TRIGGER_VERIFY` on UDK_10, confidence exactly `KWS_MATCH_CONFIDENCE`
(0.65), on the transcript `'தேங்க்ஸ்'` ("thanks") -- purely from the
audio-based Tamil KWS path, not text matching (`தேங்க்ஸ்` doesn't
textually resemble UDK_10's Tamil phrase `என்னை விடு`/"let me go" at
all). This is a genuinely new kind of finding for this project: the
first real KWS-layer (audio, not text) false positive observed on real
audio, rather than the synthetic calibration corpus. Not a surprise --
Tamil's KWS was already the second-weakest of the four Indic languages
in calibration (70% recall/25% FPR) -- but the first concrete real-audio
confirmation of it. Not yet re-tuned; stated honestly as an open,
known-weak spot rather than fixed here.

**A third real positive clip (`_converted_positives_2.wav`, a real movie
abduction scene -- Tamil dialogue, per the person providing it -- with a
loud action-movie score, 50s) surfaced a real, deeper limitation than
anything found so far: language detection can be confidently, sustained
WRONG, not just flip-floppy, and no available mitigation fixed it.**

The pipeline's live run (`tiny.en`, English, the initial guess) produced
near-total garbage transcripts and still managed to fire 3 events
(1 `TRIGGER_VERIFY` on a single word "oh", 1 `TRIGGER_VERIFY` on a loosely
related fragment, 1 `TRIGGER_ALL` via the distinct-UDK escalation rule)
-- an interesting validation that the escalation design catches a real
struggle scene's *pattern* even when individual transcripts are noise,
but NOT a clean true-positive on the actual content. Directly measuring
`VoxLingua107` against this clip found it wasn't a borderline miss: it
said `en` with 0.90-1.00 confidence on 4 of 8 six-second windows and
never once considered `ta` a candidate. This is a different failure mode
than Bug 3 above -- Bug 3 was a noisy detector randomly agreeing with
itself for 2-3 windows; this is a detector *consistently, confidently*
fooled the same wrong way the whole clip, which `LID_SWITCH_CONFIRMATIONS`
cannot fix since there's no disagreement for it to require corroboration
on.

Forcing the correct language (Tamil fine-tuned STT, `small` and then
`medium` size) did not recover clean dialogue either -- three different
model configurations (`small`-Tamil, `medium`-Tamil, generic multilingual
`small`) produced three *different* garbled results rather than
converging toward the truth, which points at a genuine acoustic
noise-floor ceiling (loud score/effects burying short shouted words),
not a fixable model-capacity gap -- the same category of honestly-stated
limitation as the existing `pitch_tempo_distorted`/overlapping-speech
conditions.

Two real, purpose-appropriate mitigations were tried and both failed,
confirming this rather than leaving it as one untested guess:
1. `speechbrain/metricgan-plus-voicebank` (speech enhancement for
   *additive stationary noise* -- fans, hiss): no change to LID (still
   `en` on every window) or STT quality. Wrong tool for a musical-score
   noise shape, diagnosed as such rather than just reported as "didn't
   work."
2. Demucs `htdemucs` vocal isolation (the actually correct tool
   category -- purpose-built to separate vocals from a music mix,
   commonly used in real audio production): **also no improvement.**
   LID stayed confidently `en` (0.62-1.00). STT stayed garbled.

A third, independent data point came for free: the Tamil STT model
produced the *same* nonsensical phrase ("so far X crore Y lakh people
have been arrested") on both the enhanced and the vocals-isolated audio
-- not real content, but the model confidently reciting what looks like
a memorized news-broadcast-style filler phrase when it can't decode the
actual audio. That's a stronger signal of a genuine confidence collapse
than plain garbling would be, and it appearing identically across two
different audio treatments is further evidence this is a real ceiling,
not a fixable configuration issue.

**Stated honestly: this is a real, now twice-confirmed gap, not a
one-off miss.** Neither dependency was added to the project
(`speechbrain`'s enhancer and `demucs` were both diagnostic-only,
installed locally for this investigation, not wired into `api.py` or
`requirements.txt`) since neither helped. Recorded here rather than
silently dropped.

## A third detection signal: acoustic scream/distress cues via pure signal processing

On request: a mathematical signal-processing signal for scream/distress
detection, explicitly NOT as a standalone verifier -- only ever adding
confidence to a segment where STT/KWS text matching has already
independently produced a real match. `scream_detector.py` is the
result: no trained model, no download, just numpy/scipy/librosa math.

**Real acoustic basis, not a guessed heuristic.** Screams are not
simply "high pitch" -- Arnal, Flinker, Kayser, Poeppel & Giraud
(*Current Biology*, 2015, "Human Screams Occupy a Privileged Niche in
the Sound Spectrum") found screams are acoustically distinguished by
**roughness**: fast amplitude modulation of the sound envelope in the
~30-150Hz range, far faster than normal speech's syllabic modulation
(~2-8Hz), and that this specifically is what the auditory system uses
to perceive urgency. The primary feature here measures that directly
(the fraction of the envelope's own modulation-spectrum power falling
in that band, via a Hilbert-envelope + FFT), combined with two weaker,
secondary features: pitch elevation above normal conversational range,
and short-term loudness.

**Real calibration corrected the initial design, not just tuned it.**
`calibrate_scream_detector.py` measured all three features separately
against RAVDESS (real human actors, fear/angry emotion at strong
intensity, n=64) vs. RAVDESS-neutral + this project's own real negative
recordings (n=78 combined), plus the one genuine real scream in this
project's corpus (the Vakeel Saab abduction clip's "Aaaah! No, no, no,
no, no!" segment, found during the real-world audio testing above).
The literature-informed starting weights were wrong in practice:
**roughness barely separated RAVDESS fear/angry from neutral at all
(0.015 vs 0.054 -- LOWER, not higher)** -- acted emotional
line-readings are not the same acoustic category as an actual scream,
the same class of mistake this project already made once treating
synthetic pitch-shift as a distress proxy (see "Closing the two
remaining honest gaps" above). **Pitch elevation turned out to be the
feature actually doing the discriminating work**: 0.045 (calm real
podcasts) -> 0.189 (RAVDESS neutral) -> 0.49-0.51 (RAVDESS fear/angry)
-> 0.381 (the real scream). Energy was actively counterproductive --
real podcast negatives scored *higher* than RAVDESS fear/angry, purely
from different recording loudness normalization between datasets, not
genuine vocal effort. Reweighted accordingly (roughness 0.15, pitch
0.80, energy 0.05) and re-swept: **0.30 threshold gives 57.8% recall /
9.0% FPR** on the combined real corpus, and correctly scores above
threshold on the real scream clip (0.320).

**57.8% recall is far too weak for a standalone trigger** -- which is
exactly why it isn't one. `udk_engine.py`'s `decide()` only even
computes/looks at `scream_score` AFTER STT/KWS has already
independently produced a TRIGGER_VERIFY-or-better match; a maxed-out
scream score with no text/KWS match at all still returns `NO_ACTION`,
unconditionally. Where it CAN act: corroborating an already-partial
match into `TRIGGER_ALL`, or overriding a `verify_by_default` phrase's
extra caution -- the same corroboration role STT+KWS agreement already
plays, just a third independent vote instead of a second. Gated behind
`UDK_ENABLE_SCREAM_DETECTION=1`, same opt-in pattern as every other
real backend here, even though this one needs no download -- it's a
new signal that can change real decisions, so it stays off by default
until a caller opts in, same guarantee every other feature here gives.

**Verified live against real audio.** With the feature enabled: the
real Vakeel Saab clip's first detected event went from a bare
`TRIGGER_VERIFY` (confidence 0.60) to `TRIGGER_ALL`, reason
`"acoustic distress signal corroborated the match"`, scream_score=0.320
-- matching the calibration measurement exactly, on a segment
containing genuine on-screen screaming. Two real negative clips that
separately produced false `TRIGGER_ALL`s did so via the pre-existing,
unrelated `"multiple distinct UDKs triggered within window"` escalation
(itself downstream of known STT-hallucination/repetition issues, not
this feature) -- their scream_scores were correctly low (0.02-0.03,
well under threshold), confirming the new signal isn't contributing
false escalations of its own.

`test_scream_detector.py` (new, 10 checks): synthetic, deterministic
DSP correctness checks (an amplitude-modulated tone inside the
roughness band scores higher than a smooth tone; elevated pitch scores
higher than normal-range pitch; silence and empty PCM both score
exactly 0.0) plus explicit corroboration-only wiring checks (a maxed
scream_score with zero text/KWS match still yields `NO_ACTION`; a
below-threshold score doesn't escalate; an at/above-threshold score
does, for both a `verify_by_default` phrase and an ordinary one at a
controlled mid-range confidence). Full regression suite: **207/207
checks passing.**

## Running it

```bash
python3 test_pipeline.py    # M1: matching/decision logic, no network needed
python3 test_m2.py          # M2: enrollment + audio storage
python3 test_m3.py          # M3: streaming API (in-process, no real network needed)
python3 test_m4.py          # M4: failure injection (duplicates, crash recovery, delivery retries)
python3 test_m5.py          # M5: durable DB, full event schema, concurrent-journey load smoke test
python3 test_m6.py          # M6: threat-model tests (auth, rate limits, encryption, signed URLs, retention)
python3 test_m7.py          # M7: metrics/SLOs, eviction leak fix, soak-shaped run, concurrency bump
python3 test_kws.py         # KWS: fusion logic, enrollment wiring, live-API STT-outage proof
python3 test_semantic.py    # Semantic: real-matcher wiring/fallback logic (fast, fake backend)
python3 test_separation.py  # Separation: NO_ACTION fallback wiring/logic (fast, fake backend)
python3 test_multilingual.py # Multilingual: normalize() fix, exact-match, live-API language wiring
python3 -c "..."            # see the pipeline smoke-test in the conversation
                             # for a full VAD->STT->Engine run

# Not part of the routine suite -- Windows-only TTS + real model inference,
# takes a few minutes, prints per-condition recall/FPR (see the section above):
python3 evaluate_pipeline_corpus.py

# Same, with the speech-separation NO_ACTION fallback and/or the tiered
# STT fallback enabled (needs speechbrain -- pip install speechbrain --
# and, for the tiered fallback, a one-time base.en download):
UDK_ENABLE_SEPARATION=1 UDK_ENABLE_TIERED_STT=1 python3 evaluate_pipeline_corpus.py

# Real-human-speech validation of the pitch/tempo distortion condition
# above (downloads the real TESS dataset, ~220MB, one-time):
python3 validate_distress_with_tess.py
python3 measure_real_distress_shift.py

# Second, independent real-human-speech dataset (RAVDESS, 24 gender-
# balanced actors, downloads individual clips via huggingface_hub):
python3 validate_distress_with_ravdess.py
python3 measure_real_distress_shift_ravdess.py

# To actually serve the API over a real socket (needs `pip install "uvicorn[standard]"`,
# not required just to run the tests above):
uvicorn api:app --reload

# Same, with the real parallel KWS path and/or real semantic matching
# enabled (needs torch+transformers+sentence-transformers, see
# requirements.txt, and kws_references.npz already generated in this checkout):
UDK_ENABLE_KWS=1 UDK_ENABLE_SEMANTIC=1 uvicorn api:app --reload

# Plus real Telugu/Kannada STT (downloads ~3GB of fine-tuned Whisper
# checkpoints on first run -- see the multilingual section above):
UDK_ENABLE_SEMANTIC=1 UDK_ENABLE_INDIC_STT=1 uvicorn api:app --reload

# Plus automatic language detection (no client-declared language at all --
# see "Automatic language detection" above; downloads faster-whisper's
# small model + speechbrain's VoxLingua107 on first run):
UDK_ENABLE_SEMANTIC=1 UDK_ENABLE_INDIC_STT=1 UDK_ENABLE_LANGUAGE_ID=1 uvicorn api:app --reload

# Plus real English STT (UDK_ENABLE_STT=1) -- without this, English
# journeys silently get MockSTT (always-empty transcripts) and can only
# ever be detected via KWS; see "Real-world audio testing" above for the
# real bug this closes. This is the full, real, all-backends-enabled config:
UDK_ENABLE_STT=1 UDK_ENABLE_SEMANTIC=1 UDK_ENABLE_INDIC_STT=1 UDK_ENABLE_INDIC_KWS=1 UDK_ENABLE_LANGUAGE_ID=1 uvicorn api:app --reload

# Plus the acoustic scream/distress corroboration signal (no download,
# pure signal processing -- see "A third detection signal" above; it
# can only ADD confidence to an already-matched segment, never trigger
# on its own):
UDK_ENABLE_STT=1 UDK_ENABLE_SEMANTIC=1 UDK_ENABLE_INDIC_STT=1 UDK_ENABLE_INDIC_KWS=1 UDK_ENABLE_LANGUAGE_ID=1 UDK_ENABLE_SCREAM_DETECTION=1 uvicorn api:app --reload

# With real infra too (needs `docker compose up -d` first):
UDK_DATABASE_URL=postgresql://udk:udk_dev_password@localhost:5432/udk \
UDK_S3_ENDPOINT=http://localhost:9000 UDK_S3_BUCKET=udk-audio \
UDK_S3_ACCESS_KEY=udk_minio UDK_S3_SECRET_KEY=udk_dev_password \
UDK_KAFKA_BOOTSTRAP_SERVERS=localhost:9092 UDK_KAFKA_TOPIC=udk-events \
uvicorn api:app --reload
```
