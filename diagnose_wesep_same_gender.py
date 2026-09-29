"""Root-cause the WeSep same-gender-overlap regression (5611f64: 7/7 direct
-> 5/7 extract-then-verify) and test whether an embedding-similarity-gated
fallback recovers it -- diagnostic + a second standalone prototype, NOT
wired into production. api.py/udk_engine.py untouched.

Reuses WeSepExtractor and the exact same clip-construction (same
phrases/voices/overlap mode) as test_wesep_target_extraction.py so the 7
same-gender and 19 different-gender clips are the identical ones already
evaluated -- Windows SAPI TTS synthesis is deterministic given the same
text+voice, so re-synthesizing reproduces the same audio bytes.

Uses speechbrain's own verify_batch() for every similarity number below
(its docstring: "Performs speaker verification with cosine distance") --
no new model, per the task's own instruction.

FINDINGS:

1. The two failures (of 7 same-gender clips) are NOT the same failure
   mode -- checked directly, not assumed:
     - Clip 1: extraction was genuinely clean (SI-SDR vs. the enrolled
       reference = +10.86dB, same order as the 5 successes; sim to the
       same-phrase clean reference = 0.4762, a real, present signal).
       Verification still rejected it (score=0.3149 < threshold=0.4111)
       against the CANONICAL enrollment sample (a different phrase from
       the one mixed) -- this is an ordinary phrase-to-phrase embedding-
       variation threshold miss on the ECAPA side, not something the
       extraction step caused. If evaluated against a same-phrase
       reference instead of the canonical enrollment clip, this clip
       would very likely pass.
     - Clip 2: a genuine extraction failure -- SI-SDR vs. enrolled is
       NEGATIVE (-4.82dB, worse than doing nothing) while SI-SDR vs. the
       INTERFERER is positive (+4.04dB), and the extracted audio's own
       embedding is far closer to the interferer (sim=0.6976) than to
       the enrolled speaker (sim=0.3356). WeSep extracted the wrong
       speaker outright.
   So of the two same-gender "failures," only ONE is actually an
   extraction problem; the other is a pre-existing ECAPA verification
   sensitivity to phrase mismatch that has nothing to do with WeSep.

2. The proposed hypothesis (enrolled<->interferer embedding similarity
   predicts which clips fail) is FALSIFIED by direct measurement, not
   assumed to hold: failures mean similarity = 0.2363 (n=2) vs.
   successes mean = 0.2656 (n=5) -- failures are, if anything, SLIGHTLY
   LOWER similarity, and the ranges overlap heavily (min failure=0.1666,
   max success=0.4448). Voice-similarity is not what's driving these two
   failures.

3. Per the task's own instruction, no similarity-gated fallback was
   built on this unsupported hypothesis -- forcing one here would be
   fitting an explanation to data that doesn't support it. Different-
   gender enrolled<->interferer similarity (mean=0.1471, max=0.2654,
   n=19) IS reliably lower than same-gender's range, consistent with the
   general "same-gender voices sound more alike" intuition -- but that
   general pattern does not, on this evidence, explain WHICH specific
   same-gender clips fail extraction.

DECISION: still not ready to propose wiring in. The 95%/71% headline
numbers from 5611f64 stand, but the mechanism behind the 71% is now
better understood (real n=2: 1 verification-threshold artifact + 1
genuine extraction miss) rather than a single clean "extraction is
weaker on same-gender voices" story. No fix identified or built --
flagged as a real, still-open question, not papered over with an
unsupported design.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from test_wesep_target_extraction import ENROLLED_VOICE, SAME_GENDER_VOICE, DIFF_GENDER_VOICE, WeSepExtractor, si_sdr
from udks import GENERAL_UDKS

SAMPLE_RATE = 16_000


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

    def sim(pcm_a: bytes, pcm_b: bytes) -> float:
        score, _ = verifier.verify_batch(pcm_to_tensor(pcm_a), pcm_to_tensor(pcm_b))
        return float(score[0])

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
        same_sims = [sim(enroll_ref, p) for p in same_speaker_calib]
        diff_sims = [sim(enroll_ref, p) for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        print(f"Verification threshold (reproduced from test_wesep_target_extraction.py): {threshold:.4f}")

        def verify(pcm: bytes) -> tuple[float, int]:
            s = sim(enroll_ref, pcm)
            return s, int(s >= threshold)

        # ---- Reconstruct the exact same 7 same-gender clips ----
        print("\nReconstructing the exact 7 same-gender-overlap clips (same phrases/voices as 5611f64)...")
        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]

        print(f"\n{'=' * 90}\nSTEP 1+2: per-clip diagnosis, all 7 same-gender clips\n{'=' * 90}")
        sg_rows = []
        for i in range(7):
            mixed = overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")
            extracted = extractor.extract(mixed, enroll_ref)

            enrolled_interferer_sim = sim(enrolled_clean[i], same_gender_unknown[i])
            extracted_vs_enrolled = sim(extracted, enrolled_clean[i])
            extracted_vs_interferer = sim(extracted, same_gender_unknown[i])
            verify_score, pred = verify(extracted)

            ref_audio = np.frombuffer(enrolled_clean[i], dtype="<i2").astype(np.float32)
            interferer_audio = np.frombuffer(same_gender_unknown[i], dtype="<i2").astype(np.float32)
            ext_audio = np.frombuffer(extracted, dtype="<i2").astype(np.float32)
            mix_audio = np.frombuffer(mixed, dtype="<i2").astype(np.float32)
            sdr_vs_enrolled = si_sdr(ext_audio, ref_audio)
            sdr_vs_interferer = si_sdr(ext_audio, interferer_audio)
            sdr_mix_vs_enrolled = si_sdr(mix_audio, ref_audio)

            outcome = "CORRECT" if pred == 1 else "WRONG"
            # Diagnosis, read from the numbers rather than assumed:
            if pred == 0:
                if extracted_vs_interferer > extracted_vs_enrolled + 0.05:
                    diagnosis = "extraction grabbed the INTERFERER, not the enrolled speaker"
                elif sdr_vs_enrolled < 0:
                    diagnosis = "extraction genuinely degraded (blend/garbled) -- SI-SDR vs enrolled is negative"
                elif verify_score > threshold - 0.15:
                    diagnosis = "extraction was reasonably clean -- verification rejected on a NORMAL threshold miss, not an extraction failure"
                else:
                    diagnosis = "extraction moderately degraded, verification correctly rejected the noisier result"
            else:
                diagnosis = "-"

            sg_rows.append({
                "i": i, "pred": pred, "verify_score": verify_score,
                "enrolled_interferer_sim": enrolled_interferer_sim,
                "sdr_vs_enrolled": sdr_vs_enrolled, "sdr_vs_interferer": sdr_vs_interferer,
                "sdr_mix_vs_enrolled": sdr_mix_vs_enrolled,
                "extracted_vs_enrolled": extracted_vs_enrolled, "extracted_vs_interferer": extracted_vs_interferer,
                "mixed": mixed, "enrolled_clean": enrolled_clean[i], "interferer": same_gender_unknown[i],
            })
            print(f"  clip {i}: {outcome:8s} verify_score={verify_score:.4f}  "
                  f"enrolled<->interferer_sim={enrolled_interferer_sim:.4f}  "
                  f"SDR(extracted,enrolled)={sdr_vs_enrolled:+.2f}dB  SDR(extracted,interferer)={sdr_vs_interferer:+.2f}dB  "
                  f"sim(extracted,enrolled)={extracted_vs_enrolled:.4f}  sim(extracted,interferer)={extracted_vs_interferer:.4f}")
            if pred == 0:
                print(f"           DIAGNOSIS: {diagnosis}")

        failures = [r for r in sg_rows if r["pred"] == 0]
        successes = [r for r in sg_rows if r["pred"] == 1]
        print(f"\n{len(failures)} failures (clips {[r['i'] for r in failures]}), {len(successes)} successes")

        print(f"\n{'=' * 90}\nSTEP 2 SUMMARY: does enrolled<->interferer similarity predict failure?\n{'=' * 90}")
        fail_sims = [r["enrolled_interferer_sim"] for r in failures]
        succ_sims = [r["enrolled_interferer_sim"] for r in successes]
        print(f"  Failures  (n={len(fail_sims)}): sims={[f'{s:.4f}' for s in fail_sims]}  mean={statistics.mean(fail_sims):.4f}" if fail_sims else "  (no failures)")
        print(f"  Successes (n={len(succ_sims)}): sims={[f'{s:.4f}' for s in succ_sims]}  mean={statistics.mean(succ_sims):.4f}" if succ_sims else "  (no successes)")
        hypothesis_holds = bool(fail_sims) and bool(succ_sims) and min(fail_sims) > max(succ_sims)
        print(f"\n  Hypothesis (failures have measurably HIGHER enrolled<->interferer similarity than successes): "
              f"{'HOLDS -- clean separation' if hypothesis_holds else 'DOES NOT CLEANLY HOLD'}")
        if not hypothesis_holds and fail_sims and succ_sims:
            print(f"  (min failure sim={min(fail_sims):.4f} vs max success sim={max(succ_sims):.4f} -- overlapping ranges)")

        # ---- Different-gender clips, for the fallback's dissimilarity check ----
        print("\nReconstructing the exact 19 different-gender-overlap clips (same phrases/voices as 5611f64)...")
        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_enrolled_interferer_sims = [sim(enrolled_clean19[i], diff_gender_unknown19[i]) for i in range(19)]
        print(f"  Different-gender enrolled<->interferer sims: mean={statistics.mean(dg_enrolled_interferer_sims):.4f} "
              f"min={min(dg_enrolled_interferer_sims):.4f} max={max(dg_enrolled_interferer_sims):.4f}")

        if not hypothesis_holds:
            print(f"\n{'=' * 90}\nSTEP 4: hypothesis does not cleanly hold -- NOT forcing a fallback design\n{'=' * 90}")
            print("  Per the task's own instruction, not building the similarity-gated fallback on unsupported")
            print("  data. See the per-clip diagnosis above for the actual (if any) pattern at this n=2 -- ")
            print("  too small to generalize from regardless of what it shows.")
            return

        # ---- Step 3: threshold determined FROM the data (midpoint), not guessed ----
        fallback_threshold = (min(fail_sims) + max(succ_sims)) / 2
        print(f"\n{'=' * 90}\nSTEP 3: similarity-gated fallback -- threshold={fallback_threshold:.4f} (midpoint of the gap above)\n{'=' * 90}")

        def combined_logic(row) -> tuple[float, int, str]:
            if row["enrolled_interferer_sim"] >= fallback_threshold:
                s, p = verify(row["mixed"])
                return s, p, "direct-verify (high similarity -> skip extraction)"
            else:
                s, p = row["verify_score"], row["pred"]
                return s, p, "extract-then-verify (low similarity)"

        print("\nSame-gender clips under combined logic:")
        sg_combined_correct = 0
        for row in sg_rows:
            s, p, path_used = combined_logic(row)
            sg_combined_correct += p
            print(f"  clip {row['i']}: sim={row['enrolled_interferer_sim']:.4f} -> {path_used} -> "
                  f"score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'})")

        print("\nDifferent-gender clips under combined logic (confirming the fallback doesn't mis-fire here):")
        dg_combined_correct = 0
        dg_fallback_triggered = 0
        for i in range(19):
            esim = dg_enrolled_interferer_sims[i]
            mixed = overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")
            if esim >= fallback_threshold:
                dg_fallback_triggered += 1
                s, p = verify(mixed)
                path_used = "direct-verify (fallback triggered)"
            else:
                extracted = extractor.extract(mixed, enroll_ref)
                s, p = verify(extracted)
                path_used = "extract-then-verify"
            dg_combined_correct += p
            print(f"  clip {i:2d}: sim={esim:.4f} -> {path_used} -> score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'})")

        print(f"\n{'=' * 90}\nBEFORE / AFTER / COMBINED-LOGIC\n{'=' * 90}")
        print(f"{'Condition':<28}{'Direct-verify (baseline)':>26}{'Extract-only (5611f64)':>26}{'Combined logic':>18}")
        print(f"{'Same-gender (n=7)':<28}{'7/7':>26}{'5/7':>26}{f'{sg_combined_correct}/7':>18}")
        print(f"{'Different-gender (n=19)':<28}{'4/19 (~21%)':>26}{'18/19 (95%)':>26}{f'{dg_combined_correct}/19 ({100*dg_combined_correct/19:.0f}%)':>18}")
        print(f"\nFallback triggered on {dg_fallback_triggered}/19 different-gender clips "
              f"({'none -- confirms different-gender embeddings stay reliably dissimilar' if dg_fallback_triggered == 0 else 'SOME -- fallback partially undermines the different-gender gain, see numbers above'})")

        print(f"\n{'=' * 90}\nREADY TO PROPOSE FOR WIRING?\n{'=' * 90}")
        ready = sg_combined_correct >= 6 and dg_combined_correct >= 17 and dg_fallback_triggered == 0
        print(f"  {'YES, with the caveats below' if ready else 'NOT YET -- see gaps above'}")
        print("  Caveat (restated, not skipped just because this is a second good result): both n=7 and")
        print("  n=19 remain small synthetic-TTS-overlap samples -- one flipped clip moves same-gender by")
        print("  ~14 points and different-gender by ~5 points. Real, evidenced signal, not a settled result.")


if __name__ == "__main__":
    main()
