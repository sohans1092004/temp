"""Two follow-ups requested after the corroboration re-evaluation
(002f17b), both standalone -- NOT wired into production
(api.py/udk_engine.py untouched, confirmed via git diff before
committing):

1. SELF-POLICING THRESHOLD CHECK: does WeSep's own extraction-confidence
   score (score_e from extract-then-verify), on its own, cleanly separate
   its known good extractions from its known bad ones -- the same way
   BEATs' calibration was found by looking directly at score
   distributions rather than gating against a second model? Reuses the
   already-checksummed positive corpus from 4775339/002f17b (92 clips,
   re-verified against verification_baseline_manifest.json here, not
   assumed from the prior run's printed output).

2. THE ACTUAL BLOCKER: everything measured in this WeSep investigation so
   far (5611f64, 1dfa86f, cebb2f0, 002f17b) has been a 100%
   enrolled-present corpus -- it only ever measured "can it find her when
   she's present," never "does it correctly say no when she's genuinely
   absent." For a safety system, a false ACCEPT (claiming the owner is
   present when she isn't) is arguably more dangerous than a false
   reject. This script builds the first NEGATIVE-ENROLLMENT test: mixture
   clips containing ONLY non-enrolled voices (enrolled speaker never
   present at all) and runs the same three-column comparison
   (direct-verify / extract-then-verify / WeSep's own score) against it.

REAL CONSTRAINT, stated honestly rather than worked around: this
machine's installed SAPI voice set (`Get-InstalledVoices`, checked
directly) has exactly 3 voices -- Zira (F, en-US, the enrolled voice),
Hazel (F, en-GB), David (M, en-US). There is no second non-enrolled
female or second non-enrolled male voice available, so a same-gender
negative-enrollment pair (two non-enrolled voices of the same gender)
cannot be constructed with this project's existing TTS setup. The
negative-enrollment overlap corpus here is necessarily a DIFFERENT-gender
pair of non-enrolled voices (Hazel + David) -- the one negative-pair
combination actually available. This is a real scope limitation of this
evaluation, not a result.

Two negative conditions, both with the enrolled speaker (Zira) NOWHERE
in the audio:
  A. SOLO non-enrolled voice, no overlap (n=5 Hazel-alone clips, reusing
     the already-checksummed `cal_diff_0..4` calibration clips from
     4775339 -- free, no new synthesis). Tests: given a single clip with
     nothing to "separate" and no genuine target, does WeSep still
     extract something and score it as a false accept?
  B. OVERLAP of two non-enrolled voices (n=19, new corpus, Hazel+David,
     NEGATIVES phrases, checksummed and saved to
     wesep_negative_enrollment_manifest.json). Tests the realistic
     product scenario: two other people talking, enrolled owner absent --
     does WeSep grab whichever voice is louder/clearer and hand it to
     verification as "the owner"?
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from test_wesep_target_extraction import ENROLLED_VOICE, SAME_GENDER_VOICE, DIFF_GENDER_VOICE, WeSepExtractor
from udks import GENERAL_UDKS

POSITIVE_MANIFEST_PATH = Path(__file__).parent / "verification_baseline_manifest.json"
NEGATIVE_MANIFEST_PATH = Path(__file__).parent / "wesep_negative_enrollment_manifest.json"


def sha256(pcm: bytes) -> str:
    return hashlib.sha256(pcm).hexdigest()


def main() -> None:
    positive_manifest = json.loads(POSITIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    positive_hashes = {c["name"]: c["sha256"] for c in positive_manifest["clips"]}
    checked_positive = []

    def positive_checksum_guard(name: str, pcm: bytes) -> bytes:
        h = sha256(pcm)
        expected = positive_hashes.get(name)
        if expected is None:
            raise RuntimeError(f"clip '{name}' not in {POSITIVE_MANIFEST_PATH.name} -- construction drifted")
        if h != expected:
            raise RuntimeError(f"clip '{name}' sha256 mismatch: got {h}, manifest has {expected}")
        checked_positive.append(name)
        return pcm

    negative_manifest: dict = {"clips": [], "results": {}}

    print("Loading WeSep extractor + ECAPA-TDNN verifier...")
    extractor = WeSepExtractor()
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_cache", local_strategy=LocalStrategy.COPY
    )

    def pcm_to_tensor(pcm: bytes) -> torch.Tensor:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0)

    def verify(enroll_ref: bytes, pcm: bytes, threshold: float) -> tuple[float, int, float]:
        t0 = time.monotonic()
        score, _ = verifier.verify_batch(pcm_to_tensor(enroll_ref), pcm_to_tensor(pcm))
        latency = time.monotonic() - t0
        score = float(score[0])
        return score, int(score >= threshold), latency

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def synth_checked(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            pcm = _load_pcm_16k_mono(p)
            return positive_checksum_guard(name, pcm)

        def synth_new(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            pcm = _load_pcm_16k_mono(p)
            negative_manifest["clips"].append({"name": name, "text": text, "voice": voice, "sha256": sha256(pcm)})
            return pcm

        phrases = [u.phrase for u in GENERAL_UDKS]

        # ---- Rebuild threshold + enrollment exactly as in 4775339/002f17b ----
        enroll_ref = synth_checked(phrases[0], ENROLLED_VOICE, "ref.wav")
        same_speaker_calib = [synth_checked(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth_checked(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [verify(enroll_ref, p, 0)[0] for p in same_speaker_calib]
        diff_sims = [verify(enroll_ref, p, 0)[0] for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        print(f"Verification threshold: {threshold:.4f}")

        def run_direct_and_extract(mixture: bytes) -> dict:
            score_d, pred_d, lat_d = verify(enroll_ref, mixture, threshold)
            t0 = time.monotonic()
            extracted = extractor.extract(mixture, enroll_ref)
            lat_extract = time.monotonic() - t0
            score_e, pred_e, lat_verify_e = verify(enroll_ref, extracted, threshold)
            return {"score_d": score_d, "pred_d": pred_d, "score_e": score_e, "pred_e": pred_e,
                    "lat_extract": lat_extract, "total_latency": lat_d + lat_extract + lat_verify_e}

        # =====================================================================
        # PART 1: reproduce the positive corpus's extract-then-verify scores
        # (checksum-verified, not reused from the prior run's printed output)
        # =====================================================================
        print(f"\n{'=' * 100}\nPART 1: re-verify WeSep's own score distribution on the KNOWN positive corpus\n{'=' * 100}")

        enrolled_clean = [synth_checked(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth_checked(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_mixtures = [positive_checksum_guard(f"sg_mix_{i}", overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")) for i in range(7)]
        sg_results = [run_direct_and_extract(m) for m in sg_mixtures]

        enrolled_clean19 = [synth_checked(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth_checked(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_mixtures = [positive_checksum_guard(f"dg_mix_{i}", overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")) for i in range(19)]
        dg_results = [run_direct_and_extract(m) for m in dg_mixtures]

        print(f"  {len(checked_positive)} clips checksum-verified against {POSITIVE_MANIFEST_PATH.name} -- byte-identical to 4775339/002f17b.")

        overlap_results = sg_results + dg_results
        extract_pass = [r for r in overlap_results if r["pred_e"] == 1]
        extract_fail = [r for r in overlap_results if r["pred_e"] == 0]
        print(f"\n  Overlap corpus (n=26): {len(extract_pass)} WeSep extract-accepts, {len(extract_fail)} WeSep extract-rejects")
        print(f"  Known-good extraction scores (extract-accept, n={len(extract_pass)}): "
              f"min={min(r['score_e'] for r in extract_pass):.4f} max={max(r['score_e'] for r in extract_pass):.4f}")
        print(f"  Known-bad extraction scores (extract-reject, n={len(extract_fail)}):  "
              f"scores={[round(r['score_e'], 4) for r in extract_fail]}")
        min_success = min(r["score_e"] for r in extract_pass)
        max_failure = max(r["score_e"] for r in extract_fail) if extract_fail else float("-inf")
        gap = min_success - max_failure
        print(f"  Gap between max-known-failure and min-known-success: {gap:.4f} "
              f"({'clean separation, current threshold ' + f'{threshold:.4f} sits inside it: ' + str(max_failure < threshold < min_success) if gap > 0 else 'NO clean gap -- scores overlap'})")

        # =====================================================================
        # PART 2A: solo non-enrolled voice, no overlap (free -- reuses cal_diff clips)
        # =====================================================================
        print(f"\n{'=' * 100}\nPART 2A: SOLO non-enrolled voice, no overlap, enrolled genuinely absent (n=5, reused cal_diff clips)\n{'=' * 100}")
        solo_neg_results = []
        for i, pcm in enumerate(diff_speaker_calib):
            t0 = time.monotonic()
            extracted = extractor.extract(pcm, enroll_ref)
            lat_extract = time.monotonic() - t0
            score_e, pred_e, _ = verify(enroll_ref, extracted, threshold)
            score_d, pred_d, _ = verify(enroll_ref, pcm, threshold)
            solo_neg_results.append({"score_d": score_d, "pred_d": pred_d, "score_e": score_e, "pred_e": pred_e, "lat_extract": lat_extract})
            print(f"  clip {i}: direct={pred_d}(score={score_d:.4f})  WeSep-extract={pred_e}(score={score_e:.4f})"
                  + ("  <-- FALSE ACCEPT" if pred_e == 1 else ""))
        solo_false_accepts = sum(1 for r in solo_neg_results if r["pred_e"] == 1)
        print(f"  Solo-negative false-accept rate (WeSep extraction): {solo_false_accepts}/{len(solo_neg_results)}")

        # =====================================================================
        # PART 2B: overlap of two non-enrolled voices, enrolled genuinely absent
        # =====================================================================
        print(f"\n{'=' * 100}\nPART 2B: OVERLAP of two non-enrolled voices (Hazel+David), enrolled genuinely absent (n=19, NEW corpus)\n{'=' * 100}")
        print("  NOTE: only one non-enrolled voice pair exists on this machine (Hazel=F/en-GB, David=M/en-US) --")
        print("  a same-gender negative pair could not be constructed. See module docstring.")

        voice_a_clips = [synth_new(NEGATIVES[i], SAME_GENDER_VOICE, f"neg_a_{i}.wav") for i in range(19)]
        voice_b_clips = [synth_new(NEGATIVES[(i + 7) % len(NEGATIVES)], DIFF_GENDER_VOICE, f"neg_b_{i}.wav") for i in range(19)]
        neg_mixtures = []
        for i in range(19):
            mixed = overlap(voice_a_clips[i], voice_b_clips[i], mode="truncate")
            negative_manifest["clips"].append({"name": f"neg_mix_{i}", "sha256": sha256(mixed)})
            neg_mixtures.append(mixed)

        neg_results = [run_direct_and_extract(m) for m in neg_mixtures]
        for i, r in enumerate(neg_results):
            flag = ""
            if r["pred_d"] == 1:
                flag += "  <-- DIRECT FALSE ACCEPT"
            if r["pred_e"] == 1:
                flag += "  <-- WESEP FALSE ACCEPT"
            print(f"  clip {i:2d}: direct={r['pred_d']}(score={r['score_d']:.3f})  "
                  f"WeSep-extract={r['pred_e']}(score={r['score_e']:.3f}){flag}")

        direct_false_accepts = sum(1 for r in neg_results if r["pred_d"] == 1)
        extract_false_accepts = sum(1 for r in neg_results if r["pred_e"] == 1)
        print(f"\n  direct-verify false-accept rate:      {direct_false_accepts}/19 ({100*direct_false_accepts/19:.0f}%)")
        print(f"  WeSep extract-then-verify false-accept rate: {extract_false_accepts}/19 ({100*extract_false_accepts/19:.0f}%)")

        negative_manifest["threshold"] = threshold
        negative_manifest["results"] = {
            "solo_negative": {"n": len(solo_neg_results), "false_accepts_extract": solo_false_accepts,
                               "scores_e": [r["score_e"] for r in solo_neg_results]},
            "overlap_negative": {"n": 19, "false_accepts_direct": direct_false_accepts, "false_accepts_extract": extract_false_accepts,
                                  "scores_d": [r["score_d"] for r in neg_results], "scores_e": [r["score_e"] for r in neg_results]},
        }
        NEGATIVE_MANIFEST_PATH.write_text(json.dumps(negative_manifest, indent=2), encoding="utf-8")
        print(f"\nSaved fixed, checksummed negative-enrollment manifest to {NEGATIVE_MANIFEST_PATH.name}")

        # =====================================================================
        # PART 3: does the self-policing threshold from Part 1 survive the negatives?
        # =====================================================================
        print(f"\n{'=' * 100}\nPART 3: does a WeSep-own-score threshold that separates Part 1's failures/successes ALSO reject Part 2's negatives?\n{'=' * 100}")
        all_negative_scores_e = [r["score_e"] for r in solo_neg_results] + [r["score_e"] for r in neg_results]
        max_negative_score = max(all_negative_scores_e)
        print(f"  Max WeSep-extraction score across ALL {len(all_negative_scores_e)} negative-enrollment clips: {max_negative_score:.4f}")
        print(f"  Min WeSep-extraction score among Part 1's known-good positive extractions: {min_success:.4f}")
        global_gap = min_success - max_negative_score
        print(f"  Gap: {global_gap:.4f} -- "
              + ("a single threshold in this gap would correctly separate EVERY known-good extraction from EVERY known-bad one, positive AND negative"
                 if global_gap > 0 else
                 "NO GAP -- at least one negative clip scored as high as a genuine positive extraction; WeSep's own score is NOT safely self-policing on its own"))

        print(f"\n{'=' * 100}\nLATENCY\n{'=' * 100}")
        all_extract_lats = [r["lat_extract"] for r in overlap_results] + [r["lat_extract"] for r in solo_neg_results] + [r["lat_extract"] for r in neg_results]
        print(f"  WeSep extraction mean latency across all {len(all_extract_lats)} clips: {statistics.mean(all_extract_lats):.3f}s")

        print(f"\n{'=' * 100}\nGO / NO-GO: WeSep-own-score-alone (no direct-verify corroboration)\n{'=' * 100}")
        recall_ok = len(extract_pass) >= round(0.90 * len(overlap_results))
        false_accept_ok = extract_false_accepts == 0 and solo_false_accepts == 0
        clean_global_gap = global_gap > 0
        print(f"  Recall on known-present positives (extract-alone, >=90% of 26): {recall_ok} ({len(extract_pass)}/26)")
        print(f"  Zero false accepts on negative-enrollment corpus (24 clips: 5 solo + 19 overlap): {false_accept_ok} "
              f"({solo_false_accepts + extract_false_accepts}/24 false accepts)")
        print(f"  A single global threshold cleanly separates ALL positives from ALL negatives: {clean_global_gap}")
        overall_go = recall_ok and false_accept_ok and clean_global_gap
        print(f"\n  RECOMMENDATION: {'GO -- propose wiring, pending sign-off' if overall_go else 'NO-GO -- see gaps above'}")


if __name__ == "__main__":
    main()
