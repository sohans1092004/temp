"""Class-specific sound-event scoring using BEATs (established in
compare_yamnet_vs_beats.py as the stronger of the two models tested).
Tests whether individual AudioSet classes actually separate the
requested categories, or collapse into the same score range. Separate,
additive investigation; does not touch the frozen guardrail work.

Real class indices used (resolved from the checkpoint's own label_dict +
Google's class_labels_indices.csv, not guessed):
  Screaming=34, Shout=344, Yell=50, Crying/sobbing=498, Groan=410,
  Speech=20, Conversation=404, Laughter=472, Chuckle/chortle=242,
  Giggle=244, Battle cry=334.
"""

from __future__ import annotations

import numpy as np

from beats_distress_detector import BEATsDistressDetector
from calibrate_scream_detector import collect_ravdess, collect_real_negatives, collect_real_scream

RELEVANT_CLASSES = {
    34: "Screaming", 344: "Shout", 50: "Yell", 498: "Crying, sobbing",
    410: "Groan", 20: "Speech", 404: "Conversation", 472: "Laughter",
    242: "Chuckle, chortle", 244: "Giggle", 334: "Battle cry",
}


def report(label: str, clips: list[bytes], detector: BEATsDistressDetector) -> None:
    all_probs = np.array([detector.label_probs(pcm) for pcm in clips])
    mean_probs = all_probs.mean(axis=0)
    max_probs = all_probs.max(axis=0)
    print(f"\n--- {label} (n={len(clips)}) ---")
    for idx, name in RELEVANT_CLASSES.items():
        print(f"  {name:20s} mean={mean_probs[idx]:.4f}  max={max_probs[idx]:.4f}")


def main() -> None:
    detector = BEATsDistressDetector()

    print("Downloading/collecting real audio for each requested class...")
    fear = collect_ravdess("fear", intensity="02")  # proxy: crying/distress speech
    angry = collect_ravdess("angry", intensity="02")  # proxy: aggressive shouting
    import calibrate_scream_detector as csd
    csd.EMOTION_CODES["happy"] = "03"
    happy = csd.collect_ravdess("happy", intensity="01")  # proxy: laughing/excited (imperfect, see report)
    negatives = collect_real_negatives()  # proxy: talking on phone (imperfect, see report)

    real_scream_full = collect_real_scream()
    # This is the (1.95, 12.93s) DSP-calibration window -- ~11s, too long
    # for a single BEATs score() call without dilution (see
    # beats_distress_detector.py's own documented gotcha). Re-slice to
    # the correctly-windowed 2s peak established earlier this session
    # (45.0-47.0s of the raw 50.5s clip) instead of reusing the long window.
    import librosa
    audio, sr = librosa.load(
        "real_recordings/positives/Nivetha Thomas Gets Abducted Vakeel Saab Malayalam Pawan Kalyan #YTShorts.mp3",
        sr=16000, mono=True,
    )
    start, end = int(45.0 * sr), int(47.0 * sr)
    single_scream_pcm = np.clip(audio[start:end] * 32767, -32768, 32767).astype(np.int16).tobytes()

    print(f"\n{'=' * 80}\nCLASS-SPECIFIC SCORING (BEATs)\n{'=' * 80}")
    report("single scream (real, correctly-windowed peak)", [single_scream_pcm], detector)
    report("crying/distress speech (RAVDESS fear, strong)", fear, detector)
    report("aggressive shouting (RAVDESS angry, strong)", angry, detector)
    report("laughing/excited (RAVDESS happy -- IMPERFECT PROXY, see note)", happy, detector)
    report("talking on phone (real negatives -- IMPERFECT PROXY, see note)", negatives[:15], detector)

    print(f"\n{'=' * 80}\nGAPS: no test audio available for these requested classes\n{'=' * 80}")
    print("  'repeated screams' (vs. a single scream) -- this project's only real scream")
    print("    clip has one identifiable peak window, not a clean repeated-scream region;")
    print("    would need dedicated sourcing (SFX library or new real recording) to test properly.")
    print("  'laughing/screaming in excitement' as a genuinely acoustically screamed/shouted")
    print("    excitement (not RAVDESS's calm read-aloud 'happy') -- RAVDESS happy above is the")
    print("    closest available real proxy, but it's normal-register happy speech, not excited")
    print("    screaming/shouting, so it likely UNDERSTATES how close this class sits to real")
    print("    'Screaming'/'Shout' scores -- flagged, not silently substituted as equivalent.")


if __name__ == "__main__":
    main()
