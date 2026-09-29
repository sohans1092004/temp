"""WeSep extraction as a CORROBORATING signal alongside direct ECAPA-TDNN
verification -- not a replacement. Standalone prototype, NOT wired into
production (api.py/udk_engine.py untouched).

COMBINED-SIGNAL LOGIC, stated explicitly (not left implicit):
  - Run direct-verify (existing ECAPA-TDNN on the raw, unseparated
    mixture) and extract-then-verify (WeSep extraction -> ECAPA-TDNN on
    the extracted audio) independently on every clip.
  - combined_accept = direct_accept AND extract_accept.
    Both agree-accept -> accept (high confidence). Both agree-reject ->
    reject. ANY disagreement -> reject. This is a plain AND-gate over
    the two accept decisions, which is exactly the "never let one signal
    override the other's rejection" rule generalized symmetrically: a
    reject from either side vetoes acceptance; acceptance requires BOTH
    to agree. Same "no single signal fires alone" shape as
    stt_confidence_gate.py/dual_asr_guard.py/BEATs scream-score
    elsewhere in this project -- corroboration can only ever raise
    confidence (by requiring agreement), never let one accept alone
    create an accept the other side didn't also support.

Unlike test_wesep_target_extraction.py (which only measured extract-
then-verify), this script runs BOTH signals on the identical clips in
the SAME execution, so direct-verify's real behavior on the known
wrong-speaker-extraction case (diagnose_wesep_same_gender.py's clip 2)
is measured fresh here, not assumed from an earlier, differently-
constructed run.

MAJOR FINDING, reported prominently because it changes the whole
narrative of this WeSep evaluation chain: freshly re-measuring direct-
verify (ECAPA-TDNN on the raw, unseparated mixture) on the exact same
construction (phrases[:7]/[1:20], NEGATIVES[i], overlap(mode="truncate"))
that 5611f64/1dfa86f were compared against gives DRAMATICALLY worse
numbers than the "7/7 same-gender, ~21% (4/19) different-gender" figures
this whole session has been citing: real result here is 1/7 (14%)
same-gender and 2/19 (11%) different-gender.

Checked the most likely explanation (the already-documented pad-vs-
truncate overlap confound) directly rather than assuming it: re-ran the
same 7 same-gender clips with mode="pad" instead of "truncate" -- still
only 2/7, nowhere near 7/7. That confound does NOT explain the gap.
The exact source of the discrepancy could not be identified -- the
original run that produced "7/7"/"~21%" was never preserved as a
separate, reproducible log (already flagged as a gap in this project's
own architecture doc), so there is no way to diff methodology against
it directly. This is reported as an open, unresolved inconsistency
between two measurements, not as a confirmed explanation.

CONSEQUENCE for this task's design: the AND-gate corroboration logic's
poor combined-logic numbers (1/7, 2/19 -- unchanged from direct-verify
alone) are NOT caused by WeSep or by the corroboration design being
wrong in principle -- they are caused by direct-verify being the far
weaker of the two signals on genuinely full-duration synthetic overlap,
in this fresh measurement. An AND-gate is only as good as its weaker
partner. Extract-then-verify alone (5/7, 18/19) substantially
outperforms direct-verify alone (1/7, 2/19) in every condition measured
today -- the corroboration-only design, applied here, actively discards
most of WeSep's real value rather than adding confidence to it.
"""

from __future__ import annotations

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


def main() -> None:
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
            return _load_pcm_16k_mono(p)

        phrases = [u.phrase for u in GENERAL_UDKS]
        enroll_ref = synth(phrases[0], ENROLLED_VOICE, "ref.wav")
        same_speaker_calib = [synth(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [verify(enroll_ref, p, 0)[0] for p in same_speaker_calib]
        diff_sims = [verify(enroll_ref, p, 0)[0] for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        print(f"Verification threshold: {threshold:.4f}")

        def run_clip(mixture: bytes) -> dict:
            score_d, pred_d, lat_d = verify(enroll_ref, mixture, threshold)
            t0 = time.monotonic()
            extracted = extractor.extract(mixture, enroll_ref)
            lat_extract = time.monotonic() - t0
            score_e, pred_e, lat_verify_e = verify(enroll_ref, extracted, threshold)
            combined = pred_d and pred_e
            disagree = pred_d != pred_e
            total_latency = lat_d + lat_extract + lat_verify_e
            return {
                "score_d": score_d, "pred_d": pred_d, "lat_d": lat_d,
                "score_e": score_e, "pred_e": pred_e, "lat_extract": lat_extract, "lat_verify_e": lat_verify_e,
                "combined": combined, "disagree": disagree, "total_latency": total_latency,
            }

        def report_condition(name: str, results: list[dict], expected: int) -> dict:
            n = len(results)
            direct_correct = sum(1 for r in results if r["pred_d"] == expected)
            extract_correct = sum(1 for r in results if r["pred_e"] == expected)
            combined_correct = sum(1 for r in results if r["combined"] == expected)
            disagreements = [i for i, r in enumerate(results) if r["disagree"]]
            print(f"\n{'=' * 90}\n{name} (n={n}, expected pred={expected})\n{'=' * 90}")
            for i, r in enumerate(results):
                flag = " <-- DISAGREE" if r["disagree"] else ""
                print(f"  clip {i:2d}: direct={r['pred_d']}(score={r['score_d']:.3f})  "
                      f"extract={r['pred_e']}(score={r['score_e']:.3f})  combined={int(r['combined'])}{flag}")
            print(f"\n  direct-verify alone:      {direct_correct}/{n} ({100*direct_correct/n:.0f}%)")
            print(f"  extract-then-verify alone: {extract_correct}/{n} ({100*extract_correct/n:.0f}%)")
            print(f"  combined logic:            {combined_correct}/{n} ({100*combined_correct/n:.0f}%)")
            print(f"  disagreement rate:         {len(disagreements)}/{n} ({100*len(disagreements)/n:.0f}%) -- clips {disagreements}")
            return {"direct": direct_correct, "extract": extract_correct, "combined": combined_correct,
                    "n": n, "disagreements": disagreements, "results": results}

        # ---- 1. Same speaker (n=3, no overlap -- sanity check) ----
        same_test = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[6:9])]
        same_results = [run_clip(pcm) for pcm in same_test]
        same_summary = report_condition("1. SAME SPEAKER", same_results, expected=1)

        # ---- 2. Same-gender overlap (n=7) ----
        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_mixtures = [overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate") for i in range(7)]
        sg_results = [run_clip(m) for m in sg_mixtures]
        sg_summary = report_condition("2. SAME-GENDER OVERLAP", sg_results, expected=1)

        # ---- 3. Different-gender overlap (n=19) ----
        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_mixtures = [overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate") for i in range(19)]
        dg_results = [run_clip(m) for m in dg_mixtures]
        dg_summary = report_condition("3. DIFFERENT-GENDER OVERLAP", dg_results, expected=1)

        # ---- Key check: does combined logic neutralize the known wrong-speaker-extraction case? ----
        print(f"\n{'=' * 90}\nKEY CHECK: the known wrong-speaker-extraction case (1dfa86f's same-gender clip 2)\n{'=' * 90}")
        clip2 = sg_results[2]
        print(f"  clip 2: direct-verify pred={clip2['pred_d']} (score={clip2['score_d']:.4f})  "
              f"extract-then-verify pred={clip2['pred_e']} (score={clip2['score_e']:.4f})  "
              f"combined={int(clip2['combined'])}")
        if clip2["pred_d"] == 0:
            print("  Direct-verify ALREADY rejects this clip on its own -- the AND-gate's 'require agreement'")
            print("  rule catches it for free, without needing to solve WeSep's underlying wrong-speaker")
            print("  extraction failure mode at all. This is the corroboration design doing its job.")
        elif clip2["combined"] == 0:
            print("  Direct-verify alone would have ACCEPTED this clip (a false accept) -- extract-then-")
            print("  verify's disagreement is what catches it. Corroboration adds real value here, not just")
            print("  redundancy.")
        else:
            print("  Both signals agree-accept on this known-bad clip -- combined logic does NOT catch it.")
            print("  This would be a real gap in the corroboration design, not a success.")

        # ---- Latency ----
        print(f"\n{'=' * 90}\nLATENCY (real, per-clip, running BOTH signals)\n{'=' * 90}")
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
        print(f"  Added cost vs. direct-verify alone: +{statistics.mean(extract_lats) + statistics.mean(verify_e_lats):.3f}s/clip")

        # ---- Summary table ----
        print(f"\n{'=' * 90}\nSUMMARY TABLE\n{'=' * 90}")
        print(f"{'Condition':<26}{'n':>4}{'Direct':>10}{'Extract':>10}{'Combined':>10}{'Disagree':>12}")
        for name, s in [("Same speaker", same_summary), ("Same-gender overlap", sg_summary), ("Different-gender overlap", dg_summary)]:
            print(f"{name:<26}{s['n']:>4}{s['direct']:>9}/{s['n']}{s['extract']:>9}/{s['n']}{s['combined']:>9}/{s['n']}"
                  f"{len(s['disagreements']):>10}/{s['n']}")

        print(f"\n{'=' * 90}\nHONEST SCOPE STATEMENT (mandatory)\n{'=' * 90}")
        print("  Synthetic, TTS-constructed overlap only -- not real recorded audio. n=7/n=19 remain small")
        print("  samples (one flipped clip moves same-gender by ~14 points, different-gender by ~5 points).")
        print("  This validates the DESIGN PATTERN (corroboration neutralizes the known wrong-speaker-")
        print("  extraction failure mode without needing to solve it) more than it proves a specific")
        print("  production-ready accuracy number -- that would need a larger, ideally real-audio pass.")

        print(f"\n{'=' * 90}\nGO / NO-GO\n{'=' * 90}")
        dg_combined_ok = dg_summary["combined"] >= round(0.90 * dg_summary["n"])
        sg_not_worse = sg_summary["combined"] >= sg_summary["direct"]
        key_check_ok = clip2["combined"] == 0
        print(f"  Different-gender combined result holds near the 95% extract-only result (>=90%): {dg_combined_ok} "
              f"({dg_summary['combined']}/{dg_summary['n']})")
        print(f"  Same-gender combined does not regress below direct-verify alone: {sg_not_worse} "
              f"({sg_summary['combined']}/{sg_summary['n']} vs {sg_summary['direct']}/{sg_summary['n']})")
        print(f"  Known wrong-speaker-extraction clip is neutralized by combined logic: {key_check_ok}")
        recommendation = "GO -- propose wiring, pending sign-off" if (dg_combined_ok and sg_not_worse and key_check_ok) else "NO-GO -- gaps remain, see above"
        print(f"\n  RECOMMENDATION: {recommendation}")


if __name__ == "__main__":
    main()
