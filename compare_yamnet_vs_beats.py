"""Direct YAMNet vs BEATs comparison on the exact same real audio used
for BEATs' own calibration this session -- same clips, same methodology,
for a fair head-to-head. Separate, additive investigation; does not
touch the frozen overlapping-speech/UDK guardrail work.
"""

from __future__ import annotations

import time

import numpy as np

from beats_distress_detector import BEATsDistressDetector
from calibrate_scream_detector import collect_ravdess, collect_real_scream, collect_real_negatives
from yamnet_distress_detector import YAMNetDistressDetector


def timed_score(detector, pcm: bytes) -> tuple[float, float]:
    t0 = time.perf_counter()
    s = detector.score(pcm)
    return s, time.perf_counter() - t0


def stats(name: str, scores: list[float]) -> None:
    arr = np.array(scores)
    print(f"  {name:28s} n={len(arr):3d} min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}")


def main() -> None:
    print("Loading BEATs...")
    beats = BEATsDistressDetector()
    print("Loading YAMNet...")
    yamnet = YAMNetDistressDetector()

    print("Extracting real scream segment (Vakeel Saab clip)...")
    real_scream = collect_real_scream()

    print("Downloading RAVDESS fear/angry (strong) + calm/happy (normal)...")
    fear = collect_ravdess("fear", intensity="02")
    angry = collect_ravdess("angry", intensity="02")
    import calibrate_scream_detector as csd
    csd.EMOTION_CODES["calm"] = "02"
    csd.EMOTION_CODES["happy"] = "03"
    calm = csd.collect_ravdess("calm", intensity="01")
    happy = csd.collect_ravdess("happy", intensity="01")

    print("Loading real negative segments (project's own real_recordings/negatives)...")
    real_negatives = collect_real_negatives()

    groups = {
        "real scream (Vakeel Saab)": [real_scream],
        "RAVDESS fear (strong)": fear,
        "RAVDESS angry (strong)": angry,
        "RAVDESS calm": calm,
        "RAVDESS happy": happy,
        "real negatives (project's own)": real_negatives,
    }

    print(f"\n{'=' * 90}\nBEATs vs YAMNet -- same clips, same methodology\n{'=' * 90}")
    beats_latencies, yamnet_latencies = [], []
    for label, clips in groups.items():
        beats_scores, yamnet_scores = [], []
        for pcm in clips:
            bs, bl = timed_score(beats, pcm)
            ys, yl = timed_score(yamnet, pcm)
            beats_scores.append(bs)
            yamnet_scores.append(ys)
            beats_latencies.append(bl)
            yamnet_latencies.append(yl)
        print(f"\n--- {label} ---")
        stats("BEATs", beats_scores)
        stats("YAMNet", yamnet_scores)

    print(f"\n{'=' * 90}\nLATENCY (per-segment inference, n={len(beats_latencies)} segments total)\n{'=' * 90}")
    ba, ya = np.array(beats_latencies), np.array(yamnet_latencies)
    print(f"  BEATs:  mean={ba.mean()*1000:.1f}ms  median={np.median(ba)*1000:.1f}ms  max={ba.max()*1000:.1f}ms")
    print(f"  YAMNet: mean={ya.mean()*1000:.1f}ms  median={np.median(ya)*1000:.1f}ms  max={ya.max()*1000:.1f}ms")
    print(f"  YAMNet is {ba.mean()/ya.mean():.1f}x faster on average" if ya.mean() > 0 else "")


if __name__ == "__main__":
    main()
