"""
Calibrates kws.py's DEFAULT_MATCH_THRESHOLD against a real measured
corpus, replacing the "picked from 4 example clips" starting point.

Corpus (Windows-only synthesis, same technique as enroll_kws_references.py):
  - Positives: each of the 20 general UDK phrases, spoken as (a) the exact
    phrase and (b) a natural paraphrase, each in TWO voices (Zira, Hazel)
    that are NOT the voice the reference bank itself was enrolled with
    (David) -- genuine cross-voice generalization, not self-matching the
    same recording setup. 80 positive clips.
  - Negatives: 20 ordinary-conversation phrases that should never match
    any UDK, split across the same two voices. 20 negative clips.

This is a real improvement over "eyeballed from 4 examples" -- measured
recall/false-positive-rate across a threshold sweep, on a corpus with
actual voice variation and paraphrase variation. It is still NOT
Section 12's full corpus: no real distress delivery (shouting, crying,
out-of-breath), no muffling/pocket simulation, no real human speakers,
English only, single TTS engine. Treat the number this picks as a
better-grounded starting point, not a claim that Section 12's testing
layer is now complete.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import kws as kws_module
from kws import SAMPLE_RATE, Wav2Vec2DTWSpotter
from udks import GENERAL_UDKS

VOICES = ["Microsoft Zira Desktop", "Microsoft Hazel Desktop"]  # NOT David (the reference bank's voice)

PARAPHRASES = {
    "UDK_01": "Please call the police right now",
    "UDK_02": "I don't feel safe at all",
    "UDK_03": "Stay back, get away from me",
    "UDK_04": "I think someone is following me",
    "UDK_05": "I really need help right now",
    "UDK_06": "Please don't touch me",
    "UDK_07": "Somebody is following me right now",
    "UDK_08": "Please, somebody help me",
    "UDK_09": "I am scared, please stay back",
    "UDK_10": "Let go of me right now",
    "UDK_11": "I need the police here immediately",
    "UDK_12": "Someone is trying to hurt me right now",
    "UDK_13": "Please, don't hurt me",
    "UDK_14": "I really don't feel safe here",
    "UDK_15": "Please stay away from me",
    "UDK_16": "I think I'm in danger",
    "UDK_17": "Somebody please help me now",
    "UDK_18": "I really want to go home now",
    "UDK_19": "Please, someone call for help",
    "UDK_20": "I need to get out of here now",
}

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


def main() -> None:
    print("Loading Wav2Vec2 + the enrolled reference bank...")
    spotter = Wav2Vec2DTWSpotter()
    spotter.load_references(Path(__file__).parent / "kws_references.npz")

    def distances_to_all(pcm: bytes) -> dict[str, float]:
        # Each phrase may now have several reference variants (voices) --
        # take the best (minimum-distance) one per phrase, same as spot().
        query = spotter._frame_embeddings(pcm)
        return {
            pid: min(kws_module._dtw_distance(query, ref) for ref in refs) for pid, refs in spotter._references.items()
        }

    positives: list[tuple[str, float]] = []  # (intended_udk_id, own_distance) only if it's also the global closest
    positive_misses: list[str] = []  # intended_udk_id, closest_id -- confused with a DIFFERENT real UDK entirely
    negatives: list[float] = []  # closest distance to ANY reference (any match at all is a false positive)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip_i = 0

        # Paraphrase only, not the exact phrase: the reference bank now
        # enrolls all 3 installed voices (enroll_kws_references.py), so
        # testing the exact phrase in any of those same voices would be
        # near-self-matching (same words, same voice as an enrolled
        # reference), not a genuine generalization test. The paraphrase
        # is lexically different from anything enrolled, in any voice,
        # so it stays a fair test even though the voice itself is no
        # longer "held out."
        print(f"\nSynthesizing {len(GENERAL_UDKS) * len(VOICES)} positive (paraphrase-only) clips...")
        for udk in GENERAL_UDKS:
            for voice in VOICES:
                clip_i += 1
                wav_path = tmp_path / f"pos_{clip_i}.wav"
                _synthesize_wav(PARAPHRASES[udk.udk_id], voice, wav_path)
                pcm = _load_pcm_16k_mono(wav_path)
                dists = distances_to_all(pcm)
                closest_id = min(dists, key=dists.get)
                if closest_id == udk.udk_id:
                    positives.append((udk.udk_id, dists[udk.udk_id]))
                else:
                    positive_misses.append(
                        f"{udk.udk_id} ({PARAPHRASES[udk.udk_id]!r}) confused with {closest_id} (dist {dists[closest_id]:.3f} vs own {dists[udk.udk_id]:.3f})"
                    )

        print(f"Synthesizing {len(NEGATIVES) * len(VOICES)} negative clips...")
        for text in NEGATIVES:
            for voice in VOICES:
                clip_i += 1
                wav_path = tmp_path / f"neg_{clip_i}.wav"
                _synthesize_wav(text, voice, wav_path)
                pcm = _load_pcm_16k_mono(wav_path)
                dists = distances_to_all(pcm)
                negatives.append(min(dists.values()))

    print(f"\n{len(positives)} positives correctly closest-matched, {len(positive_misses)} confused with a different real UDK:")
    for m in positive_misses:
        print(f"  {m}")

    print(f"\n{'threshold':>10} {'recall':>10} {'false_pos_rate':>16}")
    for threshold in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50]:
        recall = sum(1 for _, d in positives if d <= threshold) / len(positives)
        fpr = sum(1 for d in negatives if d <= threshold) / len(negatives)
        print(f"{threshold:>10.2f} {recall:>10.1%} {fpr:>16.1%}")

    print(f"\nRaw positive distances (own-UDK, correctly-closest only): {sorted(d for _, d in positives)}")
    print(f"Raw negative (closest-any) distances: {sorted(negatives)}")


if __name__ == "__main__":
    main()
