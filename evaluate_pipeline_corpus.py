"""
Section 12's test corpus, built and run for real: not a claim that real
distress recordings exist here (they don't, and Section 17 is explicit
that no dataset combining these phrases + distress delivery + real
speakers exists to find), but a genuine measured evaluation of the whole
pipeline (real webrtcvad -> real FasterWhisperSTT -> real
Wav2Vec2DTWSpotter KWS -> real UDKEngine fusion) under the specific
degraded conditions Section 12 names, built via the same signal-
augmentation technique Section 17 already recommends for KWS training
data, applied here to a broader whole-pipeline evaluation corpus instead.

Conditions covered, each derived from ONE TTS base clip via audio_augment.py
(only 40 TTS calls needed for a 140-clip corpus):
  - baseline: the base TTS clip itself, no augmentation
  - pitch_tempo_distorted: pitch up 3 semitones + 1.3x faster. NOT a
    validated distress proxy -- validate_distress_with_tess.py compared
    this against real human fear/angry speech (TESS) and found real
    emotional delivery costs only ~3-4 points of STT accuracy, while this
    synthetic distortion collapses recall to ~40-45%. The gap is a
    phase-vocoder pitch-shift artifact, not a real acoustic property of
    stressed speech -- kept as a generic "how does the pipeline handle
    heavily pitch/tempo-distorted audio" robustness check, explicitly
    relabeled from its original name ("distress_sim") once that was
    shown to be the wrong claim.
  - muffled: low-pass filtered + attenuated (pocket/bag simulation)
  - noisy: mixed with synthesized traffic-like background noise at 10dB SNR
  - overlapping: mixed with a second, different speaker's clip (real
    overlapping speech, not simulated -- two real TTS clips summed)

Explicitly NOT covered, stated plainly rather than faked: real
shouting/crying/out-of-breath delivery under genuine threat (TESS is
*acted* fear/anger, real distress could still differ), real environmental
noise recordings, multilingual/code-switching speech (all installed TTS
voices are English). This is a real, measured improvement over "0 clips
tested under degraded conditions" -- not a claim that Section 12's
corpus requirement is now fully satisfied.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from audio_augment import add_noise, muffle, overlap, pitch_shift, time_stretch
from kws import Wav2Vec2DTWSpotter
from stt import FasterWhisperSTT, TieredWhisperSTT
from udk_engine import UDKEngine, match_transcript
from udks import GENERAL_UDKS
from vad import VAD, SAMPLE_RATE

VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]

NEGATIVES = [
    "How are you doing today",
    "What a lovely day for a walk",
    "I need to go now",
    "Can you help me with this",
    "I'm really tired",
    "Call me later",
    "Let's get out of here",
    "I'm on my way",
    "See you soon",
    "What time is the meeting",
    "I love this restaurant",
    "Did you watch the game last night",
    "The weather is really nice today",
    "I need to buy some groceries",
    "Let's meet at the coffee shop",
    "My phone battery is low",
    "I'm going to bed early tonight",
    "Can you pick up some milk",
    "That movie was really funny",
    "I have a doctor's appointment tomorrow",
]


def _synthesize_wav(text: str, voice: str, out_path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech\n"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$synth.SelectVoice('{voice}')\n"
        f"$synth.SetOutputToWaveFile('{out_path}')\n"
        f"$synth.Speak('{text.replace(chr(39), chr(39) * 2)}')\n"
        "$synth.Dispose()\n"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)


def _load_pcm_16k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        n_channels, framerate = w.getnchannels(), w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if framerate != SAMPLE_RATE:
        duration = len(samples) / framerate
        src_x = np.linspace(0, duration, num=len(samples), endpoint=False)
        dst_x = np.linspace(0, duration, num=int(duration * SAMPLE_RATE), endpoint=False)
        samples = np.interp(dst_x, src_x, samples)
    return samples.astype(np.int16).tobytes()


def _run_through_pipeline(pcm: bytes, vad: VAD, stt: FasterWhisperSTT, kws: Wav2Vec2DTWSpotter, engine: UDKEngine, separator=None):
    """Runs one clip through the real pipeline exactly once, mirroring
    pipeline.py's shape (VAD -> STT+KWS in parallel -> engine fusion).
    If separator is given (UDK_ENABLE_SEPARATION=1), a NO_ACTION result
    gets one retry via separation.py's fallback -- api.py's exact wiring,
    reused here so this measures the real deployed behavior.
    Returns (decision, matched_udk_id, transcript, layer, latency_s)."""
    t0 = time.monotonic()
    segments = vad.segment_speech(pcm)
    if not segments:
        return "NO_ACTION", None, "", "none", time.monotonic() - t0
    # One clip = one utterance for this corpus; take the whole detected span.
    seg = max(segments, key=lambda s: s.end_ms - s.start_ms)
    transcript = stt.transcribe(seg.pcm)
    kws_match = kws.spot(seg.pcm)
    decision = engine.decide(transcript.text, now_s=time.monotonic(), kws_match=kws_match)
    if decision.decision == "NO_ACTION" and separator is not None:
        from separation import retry_with_separation

        recovered = retry_with_separation(seg.pcm, stt, engine, separator, time.monotonic(), kws=kws)
        if recovered is not None:
            decision = recovered
    latency = time.monotonic() - t0
    return decision.decision, (decision.udk.udk_id if decision.udk else None), transcript.text, decision.layer, latency


def main() -> None:
    print("Loading VAD, real FasterWhisperSTT, real Wav2Vec2 KWS + reference bank...")
    vad = VAD()
    if os.environ.get("UDK_ENABLE_TIERED_STT") == "1":
        print("Using TieredWhisperSTT (UDK_ENABLE_TIERED_STT=1): tiny.en with a confidence-gated base.en fallback...")
        stt = TieredWhisperSTT()
    else:
        stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")
    kws = Wav2Vec2DTWSpotter()
    kws.load_references(Path(__file__).parent / "kws_references.npz")

    separator = None
    if os.environ.get("UDK_ENABLE_SEPARATION") == "1":
        print("Loading SepFormer speech-separation fallback (UDK_ENABLE_SEPARATION=1)...")
        from separation import SepformerSeparator

        separator = SepformerSeparator()

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

        print("\nDeriving conditions and running the real pipeline on each clip...")
        results: dict[str, list[dict]] = {c: [] for c in ["baseline", "pitch_tempo_distorted", "muffled", "noisy", "overlapping"]}
        for i, udk in enumerate(GENERAL_UDKS):
            base = base_positive[udk.udk_id]
            conditions = {
                "baseline": base,
                # NOT a validated distress proxy -- see the module docstring
                # and validate_distress_with_tess.py. Real fear/angry speech
                # barely hurts STT; this synthetic distortion does not
                # represent that, it's a generic distortion-robustness check.
                "pitch_tempo_distorted": time_stretch(pitch_shift(base, SAMPLE_RATE, n_steps=3), rate=1.3),
                "muffled": muffle(base, SAMPLE_RATE),
                "noisy": add_noise(base, SAMPLE_RATE, snr_db=10, kind="traffic"),
                "overlapping": overlap(base, base_negative[i % len(base_negative)]),
            }
            for cond_name, pcm in conditions.items():
                engine = UDKEngine(GENERAL_UDKS)  # fresh engine per clip, no repetition-window cross-contamination
                decision, matched_id, transcript, layer, latency = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=separator)
                results[cond_name].append(
                    {
                        "udk_id": udk.udk_id,
                        "correct": matched_id == udk.udk_id and decision != "NO_ACTION",
                        "decision": decision,
                        "transcript": transcript,
                        "layer": layer,
                        "latency_s": latency,
                    }
                )

        neg_results: dict[str, list[dict]] = {"baseline": [], "degraded": []}
        for i, text in enumerate(NEGATIVES):
            base = base_negative[i]
            degraded = add_noise(muffle(base, SAMPLE_RATE), SAMPLE_RATE, snr_db=10, kind="crowd")
            for cond_name, pcm in [("baseline", base), ("degraded", degraded)]:
                engine = UDKEngine(GENERAL_UDKS)
                decision, matched_id, transcript, layer, latency = _run_through_pipeline(pcm, vad, stt, kws, engine, separator=separator)
                stt_match = match_transcript(transcript, GENERAL_UDKS)  # for false-positive root-cause diagnosis
                neg_results[cond_name].append(
                    {
                        "text": text,
                        "false_positive": decision != "NO_ACTION",
                        "decision": decision,
                        "transcript": transcript,
                        "layer": layer,
                        "stt_layer": stt_match.layer,
                        "stt_confidence": stt_match.confidence,
                        "latency_s": latency,
                    }
                )

    print(f"\n{'=' * 70}\nRECALL BY CONDITION (positives, n=20 each)\n{'=' * 70}")
    print(f"{'condition':<15}{'recall':>10}{'TRIGGER_ALL':>14}{'TRIGGER_VERIFY':>16}{'avg_latency_s':>16}")
    for cond, rows in results.items():
        correct = sum(1 for r in rows if r["correct"])
        trigger_all = sum(1 for r in rows if r["decision"] == "TRIGGER_ALL")
        trigger_verify = sum(1 for r in rows if r["decision"] == "TRIGGER_VERIFY")
        avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
        print(f"{cond:<15}{correct / len(rows):>10.1%}{trigger_all:>14}{trigger_verify:>16}{avg_latency:>16.3f}")

    print(f"\n{'=' * 70}\nPER-UDK PASS/FAIL MATRIX (every UDK x every condition)\n{'=' * 70}")
    conditions = list(results.keys())
    col_width = max(len(c) for c in conditions) + 2
    header = f"{'UDK':<10}" + "".join(f"{c:<{col_width}}" for c in conditions)
    print(header)
    by_udk: dict[str, dict[str, bool]] = {}
    for cond, rows in results.items():
        for r in rows:
            by_udk.setdefault(r["udk_id"], {})[cond] = r["correct"]
    for udk in GENERAL_UDKS:
        row = "".join(f"{'PASS' if by_udk[udk.udk_id][c] else 'FAIL':<{col_width}}" for c in conditions)
        print(f"{udk.udk_id:<10}{row}")

    print(f"\n{'=' * 70}\nFALSE-POSITIVE RATE BY CONDITION (negatives, n=20 each)\n{'=' * 70}")
    for cond, rows in neg_results.items():
        fpr = sum(1 for r in rows if r["false_positive"]) / len(rows)
        print(f"{cond:<15}{fpr:>10.1%}")

    print("\nFalse-positive root cause (which layer actually fired), by condition:")
    for cond, rows in neg_results.items():
        fps = [r for r in rows if r["false_positive"]]
        if fps:
            by_layer: dict[str, int] = {}
            for r in fps:
                by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + 1
            print(f"  {cond}: {by_layer}")
            for r in fps:
                print(f"    [{r['layer']}] {r['text']!r} -> transcript={r['transcript']!r} stt={r['stt_layer']}({r['stt_confidence']:.2f})")

    print("\nMisses (positives NOT correctly detected), by condition:")
    for cond, rows in results.items():
        misses = [r for r in rows if not r["correct"]]
        if misses:
            print(f"  {cond}:")
            for m in misses:
                print(f"    {m['udk_id']}: decision={m['decision']} transcript={m['transcript']!r}")


if __name__ == "__main__":
    main()
