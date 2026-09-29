"""Root-cause investigation: why did today's direct-verify (ECAPA-TDNN)
re-measurement (1/7 same-gender, 2/19 different-gender) not match the
previously-cited "7/7 same-gender, ~21% (4/19) different-gender"
baseline used throughout the WeSep evaluation chain (5611f64, 1dfa86f,
cebb2f0)? Also RE-ESTABLISHES a trustworthy baseline going forward,
saved as a fixed, versioned, checksummed artifact -- not just numbers
in a chat message -- per the task's explicit closing instruction.

FINDINGS, each checked directly rather than assumed:

1. Test-set identity could not be confirmed -- and this is itself the
   root cause. Every script that has ever computed these numbers
   (test_speaker_verification.py, test_wavlm_speaker_verification.py,
   test_wesep_target_extraction.py, test_wesep_corroboration.py,
   diagnose_wesep_same_gender.py) synthesizes into an auto-deleted
   `tempfile.TemporaryDirectory()` -- no fixed original clips were ever
   saved anywhere. A byte-for-byte checksum comparison against "the
   original" is impossible; there is nothing left to diff against.

2. THE SMOKING GUN: the code that computed "7/7 / ~21%" does not exist
   anywhere in this repository's git history. `test_speaker_verification.py`
   has exactly one commit (85d014d) and its only-ever-committed content
   is a small n=3-per-condition test plus a SINGLE-clip overlap sanity
   check (not n=7/n=19). `test_wavlm_speaker_verification.py` (also one
   commit, cee2b72) is the first and only committed appearance of the
   n=7/n=19 construction pattern -- but it only PRINTS "Compare to
   ECAPA-TDNN (today): ... 7/7 ... ~21%" as a hardcoded reference
   string; it never computes that ECAPA baseline itself. The n=7/n=19
   direct-verify baseline was therefore computed, at some point, by an
   ephemeral/uncommitted scratch script or inline computation that no
   longer exists -- not by any code this investigation can re-run or
   diff against.

3. Pipeline determinism, checked directly, is NOT the problem:
   - TTS synthesis: same text+voice re-synthesized 3x -> byte-identical
     (sha256 matched all 3 runs, verified with real hashes, not assumed).
   - ECAPA-TDNN model: self-similarity score on fixed input, 5 repeated
     calls -> 1.000000 exactly every time. No dropout-at-inference or
     other nondeterminism found.
   - audio_augment.overlap()'s "pad" branch math is byte-identical to
     before this session's changes (git diff against 0536f7b confirms
     only a NEW "truncate" branch was added; the original pad-mode
     arithmetic is untouched). Both modes were directly re-tested this
     investigation: pad=2/7, truncate=1/7 -- NEITHER reproduces 7/7,
     ruling out the pad/truncate confound as the explanation (it was
     already suspected and is explicitly ruled out here, not assumed).

4. Real, confirmed environment drift WAS found, though it cannot be
   tied definitively to the original discrepancy: torch/torchaudio were
   `2.14.0+cpu`/`2.11.0+cpu` (mismatched) earlier in this session, and
   are now `2.9.1+cpu`/`2.9.1+cpu` (matched) after installing
   `silero-vad` for the WeSep investigation pulled in a compatible
   torchaudio and downgraded torch to match. No script pins a specific
   speechbrain/spkrec-ecapa-voxceleb revision (`from_hparams(source=...)`
   with no `revision=` kwarg) -- but the local `ecapa_cache/` directory
   means the SAME cached checkpoint has been reused across all of
   today's runs regardless of version pinning, so this specific drift
   could not have silently changed the checkpoint mid-session today.
   Whether the environment differed at the time of the original,
   pre-today claim is unknowable -- no environment snapshot from that
   run was ever captured.

CONCLUSION: this is a process/provenance failure, not a pipeline
reliability problem. The pipeline is confirmed deterministic end to end.
The "7/7 / ~21%" figure was never backed by committed, reproducible
code -- it was carried forward across multiple docstrings as an
unverified claim. Every conclusion in 5611f64/1dfa86f/cebb2f0 that
described WeSep's results as a "regression" from that figure should be
read against the RE-ESTABLISHED baseline below instead: extract-then-
verify (5/7, 18/19) is a large IMPROVEMENT over the real, reproducible
direct-verify baseline (1/7, 2/19), not a marginal or regressive result.

RE-ESTABLISHED BASELINE (mode="truncate", the corrected non-confounded
overlap construction), committed here for the first time as a fixed,
checksummed, reproducible artifact -- see verification_baseline_manifest.json
for the exact sha256 of every clip and score, so this can be diffed
against in the future instead of silently drifting again:
  Same speaker (n=3):              3/3
  Same-gender overlap (n=7):       1/7  (14%)
  Different-gender overlap (n=19): 2/19 (11%)
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from test_wesep_target_extraction import ENROLLED_VOICE, SAME_GENDER_VOICE, DIFF_GENDER_VOICE
from udks import GENERAL_UDKS

MANIFEST_PATH = Path(__file__).parent / "verification_baseline_manifest.json"


def sha256(pcm: bytes) -> str:
    return hashlib.sha256(pcm).hexdigest()


def main() -> None:
    print("=" * 90)
    print("PART 1: determinism checks")
    print("=" * 90)

    print("\n1a. TTS synthesis determinism (3 repeated syntheses, same text+voice):")
    with tempfile.TemporaryDirectory() as tmp:
        hashes = []
        for i in range(3):
            p = Path(tmp) / f"det_{i}.wav"
            _synthesize_wav("Get away from me", ENROLLED_VOICE, p)
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes.append(h)
            print(f"    run {i}: sha256={h}")
        print(f"    all identical: {len(set(hashes)) == 1}")

    print("\n1b. ECAPA-TDNN model determinism (5 repeated calls, fixed input):")
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_cache", local_strategy=LocalStrategy.COPY
    )

    def pcm_to_tensor(pcm: bytes) -> torch.Tensor:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.wav"
        _synthesize_wav("Get away from me", ENROLLED_VOICE, p)
        ref_pcm = _load_pcm_16k_mono(p)
    scores = []
    for i in range(5):
        score, _ = verifier.verify_batch(pcm_to_tensor(ref_pcm), pcm_to_tensor(ref_pcm))
        scores.append(float(score[0]))
        print(f"    run {i}: self-similarity={scores[i]:.6f}")
    print(f"    all identical: {len(set(scores)) == 1}")

    print("\n" + "=" * 90)
    print("PART 2: re-establish the direct-verify baseline (mode='truncate'), fixed + checksummed")
    print("=" * 90)

    def sim(pcm_a: bytes, pcm_b: bytes) -> float:
        score, _ = verifier.verify_batch(pcm_to_tensor(pcm_a), pcm_to_tensor(pcm_b))
        return float(score[0])

    manifest: dict = {"clips": [], "results": {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def synth(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            pcm = _load_pcm_16k_mono(p)
            manifest["clips"].append({"name": name, "text": text, "voice": voice, "sha256": sha256(pcm)})
            return pcm

        phrases = [u.phrase for u in GENERAL_UDKS]
        enroll_ref = synth(phrases[0], ENROLLED_VOICE, "ref.wav")
        same_speaker_calib = [synth(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [sim(enroll_ref, p) for p in same_speaker_calib]
        diff_sims = [sim(enroll_ref, p) for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        manifest["threshold"] = threshold
        print(f"  threshold: {threshold:.4f}")

        def verify(pcm: bytes) -> tuple[float, int]:
            s = sim(enroll_ref, pcm)
            return s, int(s >= threshold)

        same_test = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[6:9])]
        same_results = [verify(pcm) for pcm in same_test]
        same_correct = sum(1 for _, p in same_results if p == 1)

        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_results = []
        for i in range(7):
            mixed = overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")
            manifest["clips"].append({"name": f"sg_mix_{i}", "sha256": sha256(mixed)})
            sg_results.append(verify(mixed))
        sg_correct = sum(1 for _, p in sg_results if p == 1)

        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_results = []
        for i in range(19):
            mixed = overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")
            manifest["clips"].append({"name": f"dg_mix_{i}", "sha256": sha256(mixed)})
            dg_results.append(verify(mixed))
        dg_correct = sum(1 for _, p in dg_results if p == 1)

        manifest["results"] = {
            "same_speaker": {"n": 3, "correct": same_correct, "scores": [s for s, _ in same_results]},
            "same_gender_overlap": {"n": 7, "correct": sg_correct, "scores": [s for s, _ in sg_results]},
            "different_gender_overlap": {"n": 19, "correct": dg_correct, "scores": [s for s, _ in dg_results]},
        }

    print(f"\n  Same speaker (n=3):              {same_correct}/3")
    print(f"  Same-gender overlap (n=7):        {sg_correct}/7  ({100*sg_correct/7:.0f}%)")
    print(f"  Different-gender overlap (n=19):  {dg_correct}/19 ({100*dg_correct/19:.0f}%)")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved fixed, checksummed baseline manifest to {MANIFEST_PATH.name}")
    print("(every synthesized clip's sha256 + every score is now on record -- re-run this script anytime")
    print(" to confirm the pipeline still reproduces byte-for-byte and score-for-score.)")


if __name__ == "__main__":
    main()
