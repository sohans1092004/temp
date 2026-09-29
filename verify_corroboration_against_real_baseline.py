"""Re-evaluate the AND-gate corroboration design (cebb2f0) against the
REAL, checksummed direct-verify baseline (4775339) instead of the
phantom "7/7 / ~21%" figure cebb2f0 was measured against. Standalone
evaluation, NOT wired into production -- api.py/udk_engine.py untouched
(confirmed via git diff before committing).

Reuses the EXACT clip construction from verify_ecapa_baseline_reliability.py
(same phrase indices, same voice constants, same overlap(mode="truncate")
call order and clip names) so this run's audio is byte-identical to the
one already checksummed in verification_baseline_manifest.json --
verified explicitly below (every clip's sha256 is checked against the
manifest before any scoring happens; a mismatch raises immediately)
rather than assumed. This is deliberately NOT a re-synthesis-from-scratch
script: the whole point of 4775339 was that regenerating audio without a
fixed reference is what let the original "7/7" discrepancy go
undiagnosed, so this evaluation must prove it's using the same clips, not
just claim to.

COMBINED-SIGNAL LOGIC (same AND-gate as cebb2f0, unchanged):
  combined_accept = direct_accept AND extract_accept. Any disagreement
  rejects. This is re-run here, not re-derived, against the real
  baseline's actual direct-verify numbers (1/7, 2/19) instead of the
  unreproducible 7/7 figure it was implicitly compared against before.

A SECOND, alternative combination rule is also evaluated per the task's
explicit request to check whether AND-gate is still the right design now
that direct-verify is confirmed to be the much weaker signal (11-14%)
against WeSep's 71-95%:

  WESEP-PRIMARY (OR-leaning): trust extract-then-verify's accept unless
  direct-verify contradicts it with HIGH confidence, not just any
  below-threshold score. "High confidence" is defined here as direct's
  score falling BELOW the calibration data's own genuine-different-
  speaker range (below max(diff_sims), the highest score any confirmed
  different-speaker calibration clip produced) -- i.e. direct-verify
  isn't just borderline-negative, it's scoring the clip as unambiguously
  a different speaker, the same category of evidence a human would call
  "confident," not "technically below the cutoff." A direct score between
  max(diff_sims) and threshold is a borderline miss (documented in
  diagnose_wesep_same_gender.py as an ECAPA phrase-mismatch sensitivity
  artifact, not necessarily a genuine different-speaker signal) and does
  NOT override WeSep under this rule.
    wesep_primary_accept = extract_accept AND NOT (direct_score < max(diff_sims))
  i.e. accept whenever WeSep accepts, unless direct-verify is confidently
  negative.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from test_wesep_target_extraction import ENROLLED_VOICE, SAME_GENDER_VOICE, DIFF_GENDER_VOICE, WeSepExtractor
from udks import GENERAL_UDKS

import hashlib
import json
import tempfile

MANIFEST_PATH = Path(__file__).parent / "verification_baseline_manifest.json"


def sha256(pcm: bytes) -> str:
    return hashlib.sha256(pcm).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_hashes = {c["name"]: c["sha256"] for c in manifest["clips"]}
    checked = []

    def checksum_guard(name: str, pcm: bytes) -> bytes:
        h = sha256(pcm)
        expected = manifest_hashes.get(name)
        if expected is None:
            raise RuntimeError(f"clip '{name}' not found in {MANIFEST_PATH.name} -- construction has drifted")
        if h != expected:
            raise RuntimeError(f"clip '{name}' sha256 mismatch: got {h}, manifest has {expected} -- NOT the same audio as 4775339")
        checked.append(name)
        return pcm

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

        def synth(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            pcm = _load_pcm_16k_mono(p)
            return checksum_guard(name, pcm)

        phrases = [u.phrase for u in GENERAL_UDKS]

        # ---- exact same construction as verify_ecapa_baseline_reliability.py ----
        enroll_ref = synth(phrases[0], ENROLLED_VOICE, "ref.wav")
        same_speaker_calib = [synth(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [verify(enroll_ref, p, 0)[0] for p in same_speaker_calib]
        diff_sims = [verify(enroll_ref, p, 0)[0] for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        confident_reject_ceiling = max(diff_sims)
        print(f"Verification threshold: {threshold:.4f}")
        print(f"'High-confidence contradiction' ceiling for WESEP-PRIMARY (max genuine-diff-speaker calib score): {confident_reject_ceiling:.4f}")

        def run_clip(mixture: bytes) -> dict:
            score_d, pred_d, lat_d = verify(enroll_ref, mixture, threshold)
            t0 = time.monotonic()
            extracted = extractor.extract(mixture, enroll_ref)
            lat_extract = time.monotonic() - t0
            score_e, pred_e, lat_verify_e = verify(enroll_ref, extracted, threshold)
            and_gate = pred_d and pred_e
            wesep_primary = pred_e and not (score_d < confident_reject_ceiling)
            disagree = pred_d != pred_e
            total_latency = lat_d + lat_extract + lat_verify_e
            return {
                "score_d": score_d, "pred_d": pred_d, "lat_d": lat_d,
                "score_e": score_e, "pred_e": pred_e, "lat_extract": lat_extract, "lat_verify_e": lat_verify_e,
                "and_gate": and_gate, "wesep_primary": int(wesep_primary), "disagree": disagree, "total_latency": total_latency,
            }

        def report_condition(name: str, results: list[dict], expected: int) -> dict:
            n = len(results)
            direct_correct = sum(1 for r in results if r["pred_d"] == expected)
            extract_correct = sum(1 for r in results if r["pred_e"] == expected)
            and_correct = sum(1 for r in results if r["and_gate"] == expected)
            wesep_primary_correct = sum(1 for r in results if r["wesep_primary"] == expected)
            disagreements = [i for i, r in enumerate(results) if r["disagree"]]
            print(f"\n{'=' * 100}\n{name} (n={n}, expected pred={expected})\n{'=' * 100}")
            for i, r in enumerate(results):
                flag = " <-- DISAGREE" if r["disagree"] else ""
                print(f"  clip {i:2d}: direct={r['pred_d']}(score={r['score_d']:.3f})  "
                      f"extract={r['pred_e']}(score={r['score_e']:.3f})  AND={int(r['and_gate'])}  WeSep-primary={r['wesep_primary']}{flag}")
            print(f"\n  direct-verify alone:      {direct_correct}/{n} ({100*direct_correct/n:.0f}%)")
            print(f"  extract-then-verify alone: {extract_correct}/{n} ({100*extract_correct/n:.0f}%)")
            print(f"  AND-gate combined:         {and_correct}/{n} ({100*and_correct/n:.0f}%)")
            print(f"  WeSep-primary combined:    {wesep_primary_correct}/{n} ({100*wesep_primary_correct/n:.0f}%)")
            print(f"  disagreement rate:         {len(disagreements)}/{n} ({100*len(disagreements)/n:.0f}%) -- clips {disagreements}")
            return {"direct": direct_correct, "extract": extract_correct, "and_gate": and_correct,
                    "wesep_primary": wesep_primary_correct, "n": n, "disagreements": disagreements, "results": results}

        # ---- 1. Same speaker (n=3, no overlap -- sanity check) ----
        same_test = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[6:9])]
        same_results = [run_clip(pcm) for pcm in same_test]
        same_summary = report_condition("1. SAME SPEAKER", same_results, expected=1)

        # ---- 2. Same-gender overlap (n=7) ----
        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_mixtures = []
        for i in range(7):
            mixed = overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")
            sg_mixtures.append(checksum_guard(f"sg_mix_{i}", mixed))
        sg_results = [run_clip(m) for m in sg_mixtures]
        sg_summary = report_condition("2. SAME-GENDER OVERLAP", sg_results, expected=1)

        # ---- 3. Different-gender overlap (n=19) ----
        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_mixtures = []
        for i in range(19):
            mixed = overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")
            dg_mixtures.append(checksum_guard(f"dg_mix_{i}", mixed))
        dg_results = [run_clip(m) for m in dg_mixtures]
        dg_summary = report_condition("3. DIFFERENT-GENDER OVERLAP", dg_results, expected=1)

        print(f"\n{'=' * 100}\nCHECKSUM VERIFICATION\n{'=' * 100}")
        print(f"  {len(checked)} clips checksum-verified against {MANIFEST_PATH.name} -- all matched, this IS the same audio as 4775339.")

        # ---- Key check: does combined logic neutralize the known wrong-speaker-extraction case? ----
        print(f"\n{'=' * 100}\nKEY CHECK: the known wrong-speaker-extraction case (1dfa86f's same-gender clip 2)\n{'=' * 100}")
        clip2 = sg_results[2]
        print(f"  clip 2: direct-verify pred={clip2['pred_d']} (score={clip2['score_d']:.4f})  "
              f"extract-then-verify pred={clip2['pred_e']} (score={clip2['score_e']:.4f})  "
              f"AND={int(clip2['and_gate'])}  WeSep-primary={clip2['wesep_primary']}")
        and_catches = clip2["and_gate"] == 0
        wesep_primary_catches = clip2["wesep_primary"] == 0
        print(f"  AND-gate catches it (rejects): {and_catches}")
        print(f"  WeSep-primary catches it (rejects): {wesep_primary_catches}"
              + ("" if wesep_primary_catches else "  <-- GAP: would NOT catch this known-bad clip"))

        # ---- Latency ----
        print(f"\n{'=' * 100}\nLATENCY (real, per-clip, running BOTH signals)\n{'=' * 100}")
        all_results = same_results + sg_results + dg_results
        direct_lats = [r["lat_d"] for r in all_results]
        extract_lats = [r["lat_extract"] for r in all_results]
        verify_e_lats = [r["lat_verify_e"] for r in all_results]
        total_lats = [r["total_latency"] for r in all_results]
        print(f"  direct-verify only:        mean={statistics.mean(direct_lats):.3f}s")
        print(f"  WeSep extraction:          mean={statistics.mean(extract_lats):.3f}s")
        print(f"  verify on extracted audio: mean={statistics.mean(verify_e_lats):.3f}s")
        print(f"  TOTAL (both signals):      n={len(total_lats)} mean={statistics.mean(total_lats):.3f}s "
              f"median={statistics.median(total_lats):.3f}s max={max(total_lats):.3f}s")

        # ---- Summary table: THIS run ----
        print(f"\n{'=' * 100}\nSUMMARY TABLE (this run, real checksummed baseline)\n{'=' * 100}")
        print(f"{'Condition':<26}{'n':>4}{'Direct':>10}{'Extract':>10}{'AND-gate':>10}{'WeSep-pri':>11}{'Disagree':>12}")
        for name, s in [("Same speaker", same_summary), ("Same-gender overlap", sg_summary), ("Different-gender overlap", dg_summary)]:
            print(f"{name:<26}{s['n']:>4}{s['direct']:>9}/{s['n']}{s['extract']:>9}/{s['n']}{s['and_gate']:>9}/{s['n']}"
                  f"{s['wesep_primary']:>10}/{s['n']}{len(s['disagreements']):>10}/{s['n']}")

        # ---- Comparison table: cebb2f0 (phantom baseline) vs. this run (real baseline) ----
        print(f"\n{'=' * 100}\nBEFORE/AFTER: cebb2f0 (phantom 7/7 baseline) vs. this run (real checksummed baseline)\n{'=' * 100}")
        direct_after_same, and_after_same = f"{same_summary['direct']}/3", f"{same_summary['and_gate']}/3"
        direct_after_sg, and_after_sg = f"{sg_summary['direct']}/7", f"{sg_summary['and_gate']}/7"
        direct_after_dg, and_after_dg = f"{dg_summary['direct']}/19", f"{dg_summary['and_gate']}/19"
        print(f"{'Condition':<26}{'Direct (before)':>18}{'Direct (after)':>18}{'AND (before)':>16}{'AND (after)':>15}")
        print(f"{'Same speaker':<26}{'3/3':>18}{direct_after_same:>18}{'3/3':>16}{and_after_same:>15}")
        print(f"{'Same-gender overlap':<26}{'1/7 (assumed 7/7 cited)':>18}{direct_after_sg:>18}{'1/7':>16}{and_after_sg:>15}")
        print(f"{'Different-gender overlap':<26}{'2/19 (assumed 4/19)':>18}{direct_after_dg:>18}{'2/19':>16}{and_after_dg:>15}")
        print("  Note: cebb2f0's OWN measured direct-verify was already 1/7, 2/19 (it discovered the discrepancy")
        print("  live) -- the 'before' column above reflects what the AND-gate design was being narratively")
        print("  compared against (the cited 7/7 figure), not what cebb2f0 itself actually measured. The AND-gate")
        print("  numbers are therefore IDENTICAL between cebb2f0 and this run (same logic, same real direct-verify")
        print("  input) -- this run's value is the checksummed reproducibility proof + the WeSep-primary alternative.")

        print(f"\n{'=' * 100}\nHONEST SCOPE STATEMENT (mandatory)\n{'=' * 100}")
        print("  Synthetic, TTS-constructed overlap only -- not real recorded audio. n=7/n=19 remain small")
        print("  samples (one flipped clip moves same-gender by ~14 points, different-gender by ~5 points).")
        print("  A corrected baseline removes the PROVENANCE problem, not this limitation -- real-audio")
        print("  validation is still needed before any of this is production-grade.")

        print(f"\n{'=' * 100}\nSTRUCTURAL QUESTION: does the AND-gate still drag WeSep down to the weak floor?\n{'=' * 100}")
        and_matches_direct_sg = sg_summary["and_gate"] == sg_summary["direct"]
        and_matches_direct_dg = dg_summary["and_gate"] == dg_summary["direct"]
        print(f"  Same-gender: AND-gate={sg_summary['and_gate']}/7 vs direct-alone={sg_summary['direct']}/7 -- "
              f"{'YES, reproduces the weak floor exactly' if and_matches_direct_sg else 'differs from direct-alone'}")
        print(f"  Different-gender: AND-gate={dg_summary['and_gate']}/19 vs direct-alone={dg_summary['direct']}/19 -- "
              f"{'YES, reproduces the weak floor exactly' if and_matches_direct_dg else 'differs from direct-alone'}")
        print(f"  WeSep-primary same-gender: {sg_summary['wesep_primary']}/7 vs extract-alone {sg_summary['extract']}/7")
        print(f"  WeSep-primary different-gender: {dg_summary['wesep_primary']}/19 vs extract-alone {dg_summary['extract']}/19")

        print(f"\n{'=' * 100}\nGO / NO-GO\n{'=' * 100}")
        and_gate_viable = (sg_summary["and_gate"] >= sg_summary["direct"]) and (dg_summary["and_gate"] >= dg_summary["direct"]) \
            and not (and_matches_direct_sg and and_matches_direct_dg)
        wesep_primary_viable = (sg_summary["wesep_primary"] >= round(0.90 * sg_summary["extract"])) \
            and (dg_summary["wesep_primary"] >= round(0.90 * dg_summary["extract"])) and wesep_primary_catches
        print(f"  AND-gate still just reproduces direct-verify's weak floor (does not add value): "
              f"{and_matches_direct_sg and and_matches_direct_dg}")
        print(f"  AND-gate recommendation: {'GO' if and_gate_viable else 'NO-GO -- still bottlenecked by the weak signal, confirmed under the real baseline too'}")
        print(f"  WeSep-primary holds near WeSep-alone accuracy (>=90% of extract-alone, both conditions): {wesep_primary_viable}")
        print(f"  WeSep-primary catches the known wrong-speaker-extraction case: {wesep_primary_catches}")
        print(f"  WeSep-primary recommendation: {'GO -- propose wiring, pending sign-off' if wesep_primary_viable else 'NO-GO'}")


if __name__ == "__main__":
    main()
