"""Real-model check of the LIVE path (api.py), for Colab: every prepped_data
clip is streamed through a real journey in 0.5 s frames (api._ingest_frame,
what the WebSocket handler calls), then the journey is finalized like
stop_journey does. Uses the evaluated best configuration's flags:
Whisper small.en + Parakeet (alert if either), KWS, semantic, BEATs, the STT
confidence gate, per-sentence matching, all on the GPU when present.

Fails loudly if any of those backends didn't really load (api.py falls back to
MockSTT / None on errors), so a silent stub can't pass as a result.

    python colab_api_check.py 2>&1 | grep -v Warning | tee api_live_check.log

Expect clip-level results close to evaluate_prepped_data.py's, not identical:
the live path segments a growing buffer frame by frame, the eval script
segments the whole clip at once.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for flag in ("UDK_ENABLE_STT", "UDK_ENABLE_PARAKEET", "UDK_ENABLE_KWS", "UDK_ENABLE_SEMANTIC",
             "UDK_ENABLE_SCREAM_DETECTION", "UDK_ENABLE_SCREAM_TRIGGER", "UDK_ENABLE_STT_CONFIDENCE_GATE"):
    os.environ.setdefault(flag, "1")

import api  # noqa: E402  (flags must be set before import: backends load at import time)
from db import JourneyDB  # noqa: E402
from event_delivery import EventDeliveryQueue  # noqa: E402
from evaluate_prepped_data import expected_udks, load_pcm, test_clips  # noqa: E402
from storage import AudioStore  # noqa: E402
from stt import MockSTT  # noqa: E402
from udks import SAMPLE_PERSONAL_UDK  # noqa: E402

FRAME_BYTES = 16000  # 0.5 s of 16 kHz int16


def main() -> None:
    print(f"device: {api._device()}")
    loaded = {
        "Whisper": not isinstance(api._stt_backend, MockSTT) and api._stt_backend,
        "Parakeet": api._parakeet_stt,
        "KWS": api._kws_backend,
        "semantic": api._semantic_matcher,
        "BEATs": api._scream_detector,
        "confidence gate": api._stt_confidence_gate_enabled,
    }
    for name, backend in loaded.items():
        print(f"  {name:<16} {'OK' if backend else 'NOT LOADED'}"
              + (f"  ({backend.model_version})" if hasattr(backend, "model_version") else "")
              + (f"  providers={backend.providers}" if hasattr(backend, "providers") else ""))
    missing = [n for n, b in loaded.items() if not b]
    if missing:
        sys.exit(f"STOP: {missing} did not load -- fix that first, a stubbed run is not a result")

    tmp = Path(tempfile.mkdtemp(prefix="udk_api_check_"))  # fresh stores, no stale journeys
    api._audio_store = AudioStore(tmp / "audio")
    api._event_queue = EventDeliveryQueue(tmp / "events")
    api._db = JourneyDB(tmp / "journeys.sqlite3")
    api._store = api.JourneyStore(db=api._db)

    outcomes, times, worst_frame = [], [], 0.0
    for i, path in enumerate(test_clips()):  # prepped_data incl. the real-traffic .mp4s
        want = expected_udks(path.name)
        journey, _ = api._store.create_or_get(
            f"check_{i}", "colab_check", api._stt_backend, SAMPLE_PERSONAL_UDK, api._audio_store,
            api._kws_backend, api._semantic_matcher,
            multilingual_semantic_matcher=api._multilingual_semantic_matcher,
            stt_by_language=api._indic_stt_backends, kws_by_language=api._indic_kws_backends,
            language_detector=api._language_detector, language_recheck_detector=api._language_recheck_detector,
        )
        pcm = load_pcm(path)
        t0 = time.monotonic()
        events = []
        for seq, start in enumerate(range(0, len(pcm), FRAME_BYTES)):
            f0 = time.monotonic()
            events += api._ingest_frame(journey, seq, pcm[start:start + FRAME_BYTES], api._audio_store)
            worst_frame = max(worst_frame, time.monotonic() - f0)
        events += api._run_incremental_detection(journey, final=True)  # what stop_journey does
        times.append(time.monotonic() - t0)

        tier = ("TRIGGER_ALL" if any(e["decision"] == "TRIGGER_ALL" for e in events)
                else "TRIGGER_VERIFY" if events else "nothing")
        outcomes.append((path.stem, bool(want), tier))
        print(f"\n{path.stem}\n  expected: {sorted(want) or 'nothing (control)'}   -> {tier}   ({times[-1]:.1f} s for {len(pcm) / 32000:.1f} s of audio)")
        for e in events:
            print(f"    {e['decision']:<15} {e['udk_id']}  conf {e['confidence']:.2f}  {e['transcript']!r}")

    tp = sum(d and t != "nothing" for _, d, t in outcomes)
    fn = sum(d and t == "nothing" for _, d, t in outcomes)
    fp = sum(not d and t != "nothing" for _, d, t in outcomes)
    tn = sum(not d and t == "nothing" for _, d, t in outcomes)
    print(f"\n{'=' * 80}\nLIVE API (api.py) RESULT")
    for stem, d, t in outcomes:
        print(f"  {'DANGER ' if d else 'control'}  {t:<15} {stem[:70]}")
    print(f"\n  Recall {tp}/{tp + fn}   FPR {fp}/{fp + tn}")
    print(f"  Missed danger clips: {[s for s, d, t in outcomes if d and t == 'nothing'] or 'none'}")
    print(f"  Time per clip: mean {sum(times) / len(times):.2f} s, max {max(times):.2f} s;"
          f" slowest single 0.5 s frame {worst_frame:.2f} s")


if __name__ == "__main__":
    main()
