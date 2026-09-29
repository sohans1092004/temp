"""
One-time (or "whenever the phrase list changes") setup: generates TTS
reference clips per general UDK phrase and saves a persisted embedding
bank via Wav2Vec2DTWSpotter.save_references(). Run this once (or whenever
GENERAL_UDKS changes), then any OS can load the resulting file at runtime
with no TTS and no Windows dependency -- generation is Windows-only
(System.Speech via PowerShell, the same technique already proven in this
project for the M1 STT verification), loading is not.

Enrolls all installed voices per phrase, not just one: spot() takes the
best (minimum-distance) reference per phrase, so more voice variants can
only help recall, never hurt it -- the tradeoff (measured, not assumed)
is a somewhat higher false-positive rate on unrelated speech, since more
references also means more chances for something unrelated to land close
to one of them by chance. See README.md for the before/after numbers.

Usage:
    python3 enroll_kws_references.py
    python3 enroll_kws_references.py --out my_references.npz
    python3 enroll_kws_references.py --personal PUDK_U1001:my secret phrase
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from kws import SAMPLE_RATE, Wav2Vec2DTWSpotter
from udks import GENERAL_UDKS

VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="kws_references.npz", help="output reference-bank path")
    parser.add_argument(
        "--personal",
        action="append",
        default=[],
        metavar="UDK_ID:PHRASE",
        help="also enroll a personal UDK, e.g. PUDK_U1001:my secret phrase",
    )
    args = parser.parse_args()

    phrases = [(udk.udk_id, udk.phrase) for udk in GENERAL_UDKS]
    for entry in args.personal:
        udk_id, _, phrase = entry.partition(":")
        phrases.append((udk_id, phrase))

    print("Loading Wav2Vec2 (first run downloads the model)...")
    spotter = Wav2Vec2DTWSpotter()

    with tempfile.TemporaryDirectory() as tmp:
        for udk_id, phrase in phrases:
            print(f"  {udk_id}: {phrase!r}")
            for voice in VOICES:
                wav_path = Path(tmp) / f"{udk_id}_{voice.split()[1]}.wav"
                _synthesize_wav(phrase, voice, wav_path)
                spotter.enroll(udk_id, _load_pcm_16k_mono(wav_path))

    spotter.save_references(args.out)
    print(f"\nSaved {len(phrases)} phrases x {len(VOICES)} voices = {len(phrases) * len(VOICES)} reference embeddings to {args.out}")


if __name__ == "__main__":
    main()
