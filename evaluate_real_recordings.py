"""
The most-real check this project can run: instead of TTS-synthesized
audio (evaluate_pipeline_corpus.py's positive set) or real human speech
that doesn't say UDK phrases (validate_distress_with_ravdess.py), this
runs ACTUAL recordings of real people saying the real UDK phrases through
the real pipeline. No dataset like this exists publicly (Section 17's
long-standing gap), so it reads from a local folder you fill in yourself.

Folder layout, both optional but at least one should have files:

    real_recordings/
        positives/   real recordings of people saying UDK phrases.
                     Filename must start with the UDK id it's meant to
                     be, e.g. UDK_01.wav, UDK_01_alice.wav,
                     UDK_08_take2.wav -- anything after the id is just a
                     label for you, ignored by this script. Record as
                     many takes/speakers per UDK as you want; UDKs with
                     zero recordings are reported as untested, not
                     assumed to pass.
        negatives/   real recordings of ordinary speech that should NOT
                     trigger anything -- any filenames, used only for
                     false-positive-rate reporting.

Audio format: 16-bit PCM WAV, any sample rate/channel count (resampled
to vad.py's SAMPLE_RATE the same way evaluate_pipeline_corpus.py does).
Convert other formats first, e.g.:
    ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 UDK_01.wav

Reuses evaluate_pipeline_corpus.py's pipeline runner and PCM loader
directly rather than re-implementing them -- the only thing that differs
from that script is where the audio comes from.
"""

from __future__ import annotations

import os
from pathlib import Path

from evaluate_pipeline_corpus import _load_pcm_16k_mono, _run_through_pipeline
from kws import Wav2Vec2DTWSpotter
from stt import FasterWhisperSTT, TieredWhisperSTT
from udk_engine import UDKEngine, match_transcript
from udks import GENERAL_UDKS, all_udks
from vad import VAD

POSITIVES_DIR = Path(__file__).parent / "real_recordings" / "positives"
NEGATIVES_DIR = Path(__file__).parent / "real_recordings" / "negatives"


def _match_udk_id_for_filename(stem: str) -> str | None:
    for udk in all_udks():
        if stem == udk.udk_id or stem.startswith(udk.udk_id + "_"):
            return udk.udk_id
    return None


def main() -> None:
    if not POSITIVES_DIR.exists() and not NEGATIVES_DIR.exists():
        print(f"Nothing to evaluate: create {POSITIVES_DIR} and/or {NEGATIVES_DIR} and add real .wav recordings.")
        print(__doc__)
        return

    print("Loading VAD, real FasterWhisperSTT, real Wav2Vec2 KWS + reference bank...")
    vad = VAD()
    if os.environ.get("UDK_ENABLE_TIERED_STT") == "1":
        print("Using TieredWhisperSTT (UDK_ENABLE_TIERED_STT=1)...")
        stt = TieredWhisperSTT()
    else:
        stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")

    kws = Wav2Vec2DTWSpotter()
    references_path = Path(__file__).parent / "kws_references.npz"
    if references_path.exists():
        kws.load_references(references_path)
    else:
        print("(no kws_references.npz found -- running without the KWS layer)")

    separator = None
    if os.environ.get("UDK_ENABLE_SEPARATION") == "1":
        print("Loading SepFormer speech-separation fallback (UDK_ENABLE_SEPARATION=1)...")
        from separation import SepformerSeparator

        separator = SepformerSeparator()

    udk_by_id = {u.udk_id: u for u in all_udks()}

    if POSITIVES_DIR.exists():
        files = sorted(POSITIVES_DIR.glob("*.wav"))
        print(f"\n{'=' * 70}\nREAL POSITIVE RECORDINGS ({len(files)} files in {POSITIVES_DIR})\n{'=' * 70}")
        if not files:
            print("  (no .wav files found)")
        tested_udks: set[str] = set()
        rows = []
        for path in files:
            expected_id = _match_udk_id_for_filename(path.stem)
            if expected_id is None:
                print(f"  [SKIP] {path.name}: filename doesn't start with a known UDK id, skipping")
                continue
            tested_udks.add(expected_id)
            pcm = _load_pcm_16k_mono(path)
            engine = UDKEngine(GENERAL_UDKS)
            decision, matched_id, transcript, layer, latency = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=separator)
            correct = matched_id == expected_id and decision != "NO_ACTION"
            rows.append((path.name, expected_id, transcript, matched_id, layer, decision, correct))
            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] {path.name} (expected {expected_id}: {udk_by_id[expected_id].phrase!r})")
            print(f"         transcript={transcript!r} matched={matched_id} layer={layer} decision={decision}")

        if rows:
            recall = sum(1 for r in rows if r[6]) / len(rows)
            print(f"\nReal-recording recall: {recall:.1%} ({sum(1 for r in rows if r[6])}/{len(rows)})")

        untested = [u.udk_id for u in GENERAL_UDKS if u.udk_id not in tested_udks]
        if untested:
            print(f"\nUDKs with NO real recording yet (untested, not assumed to pass): {', '.join(untested)}")

    if NEGATIVES_DIR.exists():
        files = sorted(NEGATIVES_DIR.glob("*.wav"))
        print(f"\n{'=' * 70}\nREAL NEGATIVE RECORDINGS ({len(files)} files in {NEGATIVES_DIR})\n{'=' * 70}")
        if not files:
            print("  (no .wav files found)")
        fp_count = 0
        for path in files:
            pcm = _load_pcm_16k_mono(path)
            engine = UDKEngine(GENERAL_UDKS)
            decision, matched_id, transcript, layer, latency = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=separator)
            stt_match = match_transcript(transcript, GENERAL_UDKS)
            is_fp = decision != "NO_ACTION"
            fp_count += is_fp
            status = "FALSE POSITIVE" if is_fp else "correct (NO_ACTION)"
            print(f"  [{status}] {path.name}: transcript={transcript!r}" + (f" -> matched={matched_id} layer={layer}({stt_match.confidence:.2f}) decision={decision}" if is_fp else ""))
        if files:
            print(f"\nReal-recording false-positive rate: {fp_count / len(files):.1%} ({fp_count}/{len(files)})")


if __name__ == "__main__":
    main()

