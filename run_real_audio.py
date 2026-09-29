"""
Run any real audio file (WAV/MP3/etc, any sample rate/channels) through
the ACTUAL production pipeline -- api.py's real _ingest_frame /
_run_incremental_detection, chunked in 500ms pieces exactly like the
live WebSocket path -- not a simplified single-utterance runner.

This is the up-to-date way to test a real recording. See also
evaluate_real_recordings.py: that script is an older, still-useful
batch labeled-corpus scorer, but predates this project's multilingual
support, language-ID, and scream detection -- it wires FasterWhisperSTT
directly (English-only, tiny.en) and does not reflect the current
production pipeline. Use THIS script for "what does the system actually
do with this one real file."

Wiring here mirrors api.py's start_journey() exactly, including two
parameters (`kws`, `semantic_matcher`) that are easy to forget and
change what "layer=semantic"/"layer=kws" in the output actually means:
match_transcript()'s semantic layer labels itself "semantic" whether a
real embedding model was given or not -- with no semantic_matcher, it
silently falls back to a rapidfuzz-based stub, still labeled
"layer=semantic" in every printed result. Passing api._semantic_matcher
here is what makes an English result's "layer=semantic" mean the real
sentence-transformers model, not the stub.

Usage:
    python3 run_real_audio.py path/to/recording.mp3
    python3 run_real_audio.py path/to/recording.mp3 --verbose   # per-segment trace incl. VAD/LID

By default enables every real opt-in backend (English STT, semantic
matching, Indic STT, Indic KWS, language-ID, scream detection) unless
you've already set the corresponding UDK_ENABLE_* env var yourself
before running this -- matches README.md's "full, real, all-backends-
enabled config". Heavy models load on first run (may take a minute or
two); a fresh, unique journey_id is used every run specifically to avoid
a real gotcha found this project: reusing a deterministic journey_id
across runs can silently replay stale state from the persistent
SQLite DB / AudioStore instead of processing the file fresh.
"""

from __future__ import annotations

import os
import sys

_DEFAULT_FLAGS = [
    "UDK_ENABLE_STT",
    "UDK_ENABLE_SEMANTIC",
    "UDK_ENABLE_INDIC_STT",
    "UDK_ENABLE_INDIC_KWS",
    "UDK_ENABLE_LANGUAGE_ID",
    "UDK_ENABLE_SCREAM_DETECTION",
]
for _flag in _DEFAULT_FLAGS:
    os.environ.setdefault(_flag, "1")

import uuid
from pathlib import Path

import librosa
import numpy as np

import api

SAMPLE_RATE = 16_000
CHUNK_MS = 500
CHUNK_BYTES = int(SAMPLE_RATE * (CHUNK_MS / 1000.0)) * 2


def load_pcm(path: Path) -> bytes:
    """librosa decodes MP3/WAV/etc and resamples in one step -- no
    ffmpeg needed (libsndfile handles MP3 directly on this platform)."""
    audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return audio_i16.tobytes()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    path = Path(args[0])
    verbose = "--verbose" in sys.argv

    print(f"Enabled backends: {', '.join(f for f in _DEFAULT_FLAGS if os.environ.get(f) == '1') or '(none)'}")

    if verbose:
        _install_trace()

    pcm = load_pcm(path)
    duration_s = len(pcm) / 2 / SAMPLE_RATE
    print(f"Loaded {path.name}: {duration_s:.1f}s\n")

    # Fresh id every run, on purpose -- see module docstring.
    journey_id = f"J_manual_{path.stem}_{uuid.uuid4().hex[:8]}"
    journey, _ = api._store.create_or_get(
        journey_id,
        "U_manual_test",
        api._stt_backend,
        api.SAMPLE_PERSONAL_UDK,
        api._audio_store,
        api._kws_backend,
        api._semantic_matcher,
        multilingual_semantic_matcher=api._multilingual_semantic_matcher,
        stt_by_language=api._indic_stt_backends,
        kws_by_language=api._indic_kws_backends,
        language_detector=api._language_detector,
        language_recheck_detector=api._language_recheck_detector,
    )

    seq = 0
    for i in range(0, len(pcm), CHUNK_BYTES):
        api._ingest_frame(journey, seq, pcm[i : i + CHUNK_BYTES], api._audio_store)
        seq += 1
    api._run_incremental_detection(journey, final=True)

    print(f"\nDetected language: {journey.language}")
    if journey.events:
        print(f"\n{len(journey.events)} event(s) fired:")
        for ev in journey.events:
            print(f"  [{ev['decision']}] udk={ev.get('udk_id')} conf={ev.get('confidence')} transcript={ev.get('transcript')!r}")
    else:
        print("\nNo events fired (NO_ACTION throughout).")


def _install_trace() -> None:
    from udk_engine import UDKEngine

    orig_decide = UDKEngine.decide

    def traced(self, transcript, now_s=None, kws_match=None, scream_score=None, **kwargs):
        result = orig_decide(self, transcript, now_s=now_s, kws_match=kws_match, scream_score=scream_score, **kwargs)
        if transcript.strip() or kws_match is not None:
            extra = f" {kwargs}" if kwargs else ""
            print(
                f"  [segment] decision={result.decision} conf={result.confidence:.2f} "
                f"layer={result.layer} scream_score={scream_score}{extra} transcript={transcript!r}"
            )
        return result

    UDKEngine.decide = traced

    orig_update = api._maybe_update_language

    def traced_update(journey):
        lang_before = journey.language
        orig_update(journey)
        if journey.language != lang_before:
            print(f"  [language] {lang_before} -> {journey.language}")

    api._maybe_update_language = traced_update


if __name__ == "__main__":
    main()
