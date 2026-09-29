"""The "join check" requested this session: does beats_distress_detector's
scream_score already discriminate on the overlapping-condition failures
that separation.py's SepFormer fallback does NOT recover (the "7
unrecovered" of the "10 real overlap failures reproduced from the corpus"
in separation.py's docstring)? If scream_score is not elevated on those
specific segments, a new ACOUSTIC_VERIFY escalation path (proposed in
conversation) would have no signal to work with for this exact gap --
this check has to come before any threshold/path design.

Reuses evaluate_pipeline_corpus.py's exact synthesis (Windows SAPI TTS)
and pipeline-running functions rather than reimplementing them, so this
measures the real deployed shape, same corpus-generation method that
originally produced the "10 failures" figure.

Note: evaluate_pipeline_corpus.py's _run_through_pipeline() does not pass
scream_score into engine.decide() at all -- it predates beats_distress_
detector.py. So the NO_ACTION failures found here are identical in kind
to the ones separation.py's docstring numbers came from (found without
any acoustic signal in the loop), and scream_score is computed here
SEPARATELY, after the fact, purely to answer the join question -- not
fed into decide() (that would be a design change, not a measurement).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from evaluate_pipeline_corpus import _load_pcm_16k_mono, _run_through_pipeline, _synthesize_wav, NEGATIVES, VOICES
from kws import Wav2Vec2DTWSpotter
from stt import FasterWhisperSTT
from udk_engine import UDKEngine
from udks import GENERAL_UDKS
from vad import VAD, SAMPLE_RATE
from audio_augment import overlap


def main() -> None:
    print("Loading VAD, real FasterWhisperSTT, real Wav2Vec2 KWS + reference bank...")
    vad = VAD()
    stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")
    kws = Wav2Vec2DTWSpotter()
    kws.load_references(Path(__file__).parent / "kws_references.npz")

    print("Loading real SepFormer separator (separation.py's actual deployed fallback)...")
    from separation import SepformerSeparator

    separator = SepformerSeparator()

    print("Loading real BEATs distress detector (beats_distress_detector.py)...")
    from beats_distress_detector import BEATsDistressDetector

    beats = BEATsDistressDetector()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print(f"\nSynthesizing {len(GENERAL_UDKS)} base positive clips...")
        base_positive: dict[str, bytes] = {}
        for i, udk in enumerate(GENERAL_UDKS):
            voice = VOICES[i % len(VOICES)]
            wav_path = tmp_path / f"pos_{udk.udk_id}.wav"
            _synthesize_wav(udk.phrase, voice, wav_path)
            base_positive[udk.udk_id] = _load_pcm_16k_mono(wav_path)

        print(f"Synthesizing {len(NEGATIVES)} base negative clips...")
        base_negative: list[bytes] = []
        for i, text in enumerate(NEGATIVES):
            voice = VOICES[i % len(VOICES)]
            wav_path = tmp_path / f"neg_{i}.wav"
            _synthesize_wav(text, voice, wav_path)
            base_negative.append(_load_pcm_16k_mono(wav_path))

        print("\nBuilding the 'overlapping' condition (same method as evaluate_pipeline_corpus.py) "
              "and running the real pipeline WITHOUT separation first...")
        raw_failures = []  # (udk_id, pcm, seg_pcm)
        for i, udk in enumerate(GENERAL_UDKS):
            base = base_positive[udk.udk_id]
            pcm = overlap(base, base_negative[i % len(base_negative)])
            engine = UDKEngine(GENERAL_UDKS)
            decision, matched_id, transcript, layer, _ = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=None)
            correct = matched_id == udk.udk_id and decision != "NO_ACTION"
            status = "OK  " if correct else "FAIL"
            print(f"  [{status}] {udk.udk_id:8s} {udk.phrase!r:32s} -> decision={decision} matched={matched_id} transcript={transcript!r}")
            if not correct:
                segments = vad.segment_speech(pcm)
                seg = max(segments, key=lambda s: s.end_ms - s.start_ms) if segments else None
                raw_failures.append((udk.udk_id, udk.phrase, pcm, seg))

        print(f"\n{len(raw_failures)} raw overlapping-condition failures (no separation, no scream_score) "
              f"out of {len(GENERAL_UDKS)} -- separation.py's docstring reports 10 from an earlier run of this "
              f"same measurement; this run's exact count may differ slightly (TTS/negative-pairing is deterministic "
              f"here, but model versions may have shifted since).")

        print(f"\n{'=' * 70}\nApplying separation.py's real fallback to each raw failure\n{'=' * 70}")
        unrecovered = []
        for udk_id, phrase, pcm, seg in raw_failures:
            engine = UDKEngine(GENERAL_UDKS)
            decision, matched_id, transcript, layer, _ = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=separator)
            recovered = matched_id == udk_id and decision != "NO_ACTION"
            print(f"  {udk_id:8s} {phrase!r:32s} separation_recovered={recovered} (decision={decision}, transcript={transcript!r})")
            if not recovered:
                unrecovered.append((udk_id, phrase, seg))

        print(f"\n{len(raw_failures) - len(unrecovered)}/{len(raw_failures)} recovered by separation "
              f"(separation.py's docstring reports 3/10 from the original measurement).")

        print(f"\n{'=' * 70}\nTHE JOIN CHECK: BEATs scream_score on the {len(unrecovered)} UNRECOVERED segments\n{'=' * 70}")
        print("(SCREAM_SCORE_THRESHOLD = 0.30, from beats_distress_detector.py's real calibration; "
              "real negatives measured this session topped out at 0.018)")
        scores = []
        for udk_id, phrase, seg in unrecovered:
            if seg is None:
                print(f"  {udk_id:8s} {phrase!r:32s} no VAD segment found, skipping")
                continue
            score = beats.score(seg.pcm)
            scores.append(score)
            elevated = "ELEVATED" if score >= 0.30 else ("some signal" if score >= 0.05 else "flat/near-zero")
            print(f"  {udk_id:8s} {phrase!r:32s} scream_score={score:.4f}  ({elevated})")

        if scores:
            print(f"\nscream_score on unrecovered failures: min={min(scores):.4f} max={max(scores):.4f} "
                  f"mean={sum(scores)/len(scores):.4f}")
            n_elevated = sum(1 for s in scores if s >= 0.30)
            print(f"{n_elevated}/{len(scores)} already clear SCREAM_SCORE_THRESHOLD (0.30) on their own.")
        else:
            print("No unrecovered failures had a usable VAD segment -- nothing to score.")


if __name__ == "__main__":
    main()
