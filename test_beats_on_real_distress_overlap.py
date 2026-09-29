"""Re-test BEATs acoustic-only escalation (beats_distress_detector.py)
against REAL distress-delivery audio (RAVDESS fear/angry, strong
intensity), not the synthetic TTS-overlap corpus used earlier this
session. That corpus (evaluate_pipeline_corpus.py's "overlapping"
condition) has zero vocal strain/screaming/crying by construction --
this test asks whether the earlier flat scream_score result (mean
0.0039, max 0.0114 on 8 unrecovered failures) was a real finding about
BEATs, or an artifact of testing it on audio with no distress delivery
at all.

Reuses calibrate_scream_detector.py's exact RAVDESS download logic
(same REPO_ID, actors, filenames) and evaluate_pipeline_corpus.py's
exact negative-clip synthesis + audio_augment.overlap() function, for
direct comparability to everything already measured this session.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from audio_augment import overlap
from calibrate_scream_detector import collect_ravdess
from evaluate_pipeline_corpus import NEGATIVES, VOICES, _load_pcm_16k_mono, _synthesize_wav

SCREAM_SCORE_THRESHOLD = 0.30
SYNTHETIC_CORPUS_MEAN = 0.0039  # 8 unrecovered TTS-overlap failures, from earlier this session
SYNTHETIC_CORPUS_MAX = 0.0114
REAL_NEGATIVE_CEILING = 0.018  # 6 real negative clips, from earlier this session


def main() -> None:
    from beats_distress_detector import BEATsDistressDetector

    detector = BEATsDistressDetector()

    print("Downloading RAVDESS fear (strong intensity, real human actors)...")
    fear_clips = collect_ravdess("fear", intensity="02")
    print(f"  got {len(fear_clips)} fear clips")
    print("Downloading RAVDESS angry (strong intensity, secondary check)...")
    angry_clips = collect_ravdess("angry", intensity="02")
    print(f"  got {len(angry_clips)} angry clips")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"\nSynthesizing {len(NEGATIVES)} negative clips (same pool as evaluate_pipeline_corpus.py)...")
        negatives = []
        for i, text in enumerate(NEGATIVES):
            voice = VOICES[i % len(VOICES)]
            wav_path = tmp_path / f"neg_{i}.wav"
            _synthesize_wav(text, voice, wav_path)
            negatives.append(_load_pcm_16k_mono(wav_path))

        def evaluate(label: str, distress_clips: list[bytes]):
            clean_scores = [detector.score(pcm) for pcm in distress_clips]
            overlaid_scores = []
            for i, pcm in enumerate(distress_clips):
                mixed = overlap(pcm, negatives[i % len(negatives)])
                overlaid_scores.append(detector.score(mixed))
            return clean_scores, overlaid_scores

        print(f"\nScoring BEATs on clean + overlaid fear clips (n={len(fear_clips)})...")
        fear_clean, fear_overlaid = evaluate("fear", fear_clips)
        print(f"Scoring BEATs on clean + overlaid angry clips (n={len(angry_clips)})...")
        angry_clean, angry_overlaid = evaluate("angry", angry_clips)

    def stats(name: str, scores: list[float]):
        arr = np.array(scores)
        n_over = int(np.sum(arr >= SCREAM_SCORE_THRESHOLD))
        print(f"  {name:30s} n={len(arr):3d}  min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}  "
              f"{n_over}/{len(arr)} clear {SCREAM_SCORE_THRESHOLD} threshold ({100*n_over/len(arr):.0f}%)")

    print(f"\n{'=' * 78}\nRESULTS\n{'=' * 78}")
    stats("RAVDESS fear, CLEAN (unoverlaid)", fear_clean)
    stats("RAVDESS fear, OVERLAID with negative", fear_overlaid)
    stats("RAVDESS angry, CLEAN (unoverlaid)", angry_clean)
    stats("RAVDESS angry, OVERLAID with negative", angry_overlaid)
    print(f"\n  [reference] synthetic TTS-overlap corpus (8 unrecovered failures): mean={SYNTHETIC_CORPUS_MEAN} max={SYNTHETIC_CORPUS_MAX}")
    print(f"  [reference] real-negative ceiling (6 real negative clips):         max={REAL_NEGATIVE_CEILING}")

    n_total = len(fear_clips) + len(angry_clips)
    print(f"\n{'=' * 78}\nSAMPLE SIZE CHECK\n{'=' * 78}")
    if n_total < 15:
        print(f"  *** n={n_total} total distress clips (fear={len(fear_clips)}, angry={len(angry_clips)}) is BELOW the "
              f"requested n>=15-20 minimum. Any conclusion below is PRELIMINARY, not a settled result, per this "
              f"session's own standard (forced-scoring and dual-ASR findings were only trusted after n>=15-20). ***")
    else:
        print(f"  n={n_total} total distress clips (fear={len(fear_clips)}, angry={len(angry_clips)}) meets the "
              f"requested n>=15-20 minimum.")

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    overlaid_all = fear_overlaid + angry_overlaid
    max_overlaid = max(overlaid_all)
    mean_overlaid = float(np.mean(overlaid_all))
    if max_overlaid > REAL_NEGATIVE_CEILING * 3:  # a real, non-marginal separation from the negative ceiling
        print(f"  Real distress delivery, overlaid, scores meaningfully above the real-negative ceiling "
              f"({REAL_NEGATIVE_CEILING}): max={max_overlaid:.4f}, mean={mean_overlaid:.4f}. "
              f"This DOES NOT confirm the earlier null result generalizes -- the synthetic-corpus flat "
              f"result may have been corpus-specific after all. The ACOUSTIC_VERIFY ruling should be "
              f"reconsidered for real (not synthetic) overlapping distress audio, pending the n-size caveat above.")
    else:
        print(f"  Real distress delivery, overlaid, remains near the real-negative ceiling "
              f"({REAL_NEGATIVE_CEILING}): max={max_overlaid:.4f}, mean={mean_overlaid:.4f}. "
              f"This CONFIRMS the earlier null result generalizes beyond the synthetic corpus -- BEATs shows "
              f"no usable acoustic-only signal on real distress-under-overlap audio either. The ACOUSTIC_VERIFY "
              f"ruling (no signal in either corpus -> don't build) stands.")


if __name__ == "__main__":
    main()
