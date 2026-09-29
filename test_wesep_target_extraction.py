"""WeSep target-speaker extraction as a fix for the different-gender-
overlap verification gap -- standalone evaluation, NOT wired into any
production code (api.py, udk_engine.py untouched).

REAL, non-obvious setup finding, documented because it cost most of this
evaluation's effort and would trip up any future attempt to reuse WeSep:
the ONLY publicly-hosted pretrained checkpoint (ModelScope,
bsrnn_ecapa_vox1.tar.gz, referenced by wesep's own cli/hub.py) was
trained against a v1, flat-config BSRNN class ("model.tse_model: BSRNN",
config_args as top-level keys like spk_model/win/stride/spk_emb_dim).
The current wenet-e2e/wesep `main` branch has since been refactored to a
v2 "modular multi-cue" architecture (commit 8c2fc21, "introduce modular
multi-cue WeSep v0.1") that renamed this class to TSE_BSRNN_SPK and
changed its config schema to nested dicts (config['separator'],
config['speaker']). Loading the real, currently-downloadable checkpoint
through the current `main` branch's Extractor class fails outright
(KeyError: Unknown model: BSRNN) -- the released checkpoint and the
released code are for two different, incompatible schema versions of
the same project. Confirmed by checking out the exact pre-refactor
commit (c5e9ce1, `git log -S"class BSRNN"`) and constructing that
version's BSRNN class directly from the checkpoint's own config.yaml --
loads with ZERO missing/unexpected keys (verified), unlike the KeyError
from `main`.

Getting from there to a runnable extraction also required bypassing two
unrelated heavy eager-import chains (wesep's own __init__.py pulling in
its full CLI + silero-vad; wespeaker's __init__.py eagerly importing
every speaker-model family including optional s3prl/whisper frontends
that have nothing to do with the one ECAPA-TDNN model this checkpoint
uses) via lightweight sys.modules stubs -- see WeSepExtractor.__init__
below. And bypassing torchaudio.load() specifically (this project's
already-documented torchcodec incompatibility, see
Model-snr/overlap_gate.py's docstring) by loading audio via librosa
instead, exactly as elsewhere in this codebase.

The exact enrollment-embedding preprocessing (fbank + CMVN, not raw
waveform or a precomputed embedding vector, since this checkpoint was
trained with model_args.tse_model.spk_feat=true) was reverse-engineered
from wesep's own dataset/processor.py (compute_fbank + apply_cmvn) at
the same pre-refactor commit -- kaldi-style fbank (80 mel bins, 25ms/10ms
frame length/shift, dither=1.0, hamming window) on the enrollment
waveform scaled by 2**15, then per-utterance mean-normalized. Note
dither=1.0 (a real training-time value, not overridden for inference in
the original test script either) introduces a small amount of run-to-run
stochasticity in the enrollment embedding -- a real, inherent property
of this checkpoint's expected input, not a bug in this evaluation.

VERDICT: MIXED -- clears the target-metric bar decisively, but does not
cleanly clear the "holding steady" condition. Real numbers (synthetic
TTS overlap, mode="truncate", same corpus/voices as today's ECAPA/WavLM
tests):

  Same speaker (direct verify, n=3):                    3/3   (sanity check, unaffected)
  Same-gender overlap (extract-then-verify, n=7):        5/7  (71%) -- REGRESSION from
                                                                direct-verify's existing 7/7 (100%)
  Different-gender overlap (extract-then-verify, n=19): 18/19 (95%) -- vs. direct-verify's
                                                                ~21% (4/19). Clears the 60%
                                                                bar with a huge margin.
  Extraction quality (SI-SDRi vs the known clean enrolled reference):
    same-gender overlap:      mean +9.44dB  (min -2.21dB -- one clip's extraction was
                                              WORSE than the raw mixture)
    different-gender overlap: mean +13.76dB (min +10.12dB -- consistently strong)
  Latency (CPU, extraction only): n=26 mean=0.85s median=0.83s max=1.06s -- fast,
    real-time-compatible, on top of the existing ECAPA verify call.

ROOT CAUSE of the same-gender regression, read directly from the SI-SDR
numbers, not guessed: extraction quality is measurably weaker and more
variable on same-gender overlap (mean +9.44dB, one negative case) than
on different-gender overlap (mean +13.76dB, uniformly strong) -- the
model's speaker-embedding conditioning has a harder time separating two
acoustically-similar (same-gender) voices, an intuitive and explainable
limitation, not a pipeline bug. This is exactly the diagnosis the task
asked for: extraction quality itself explains the downstream verification
regression, not a separate verification-side problem.

DECISION, per the task's explicit bar ("different-gender-overlap >= ~60%
WITH same-speaker/same-gender-overlap holding steady"): the target metric
clears its bar by a wide margin (95% vs. 60%), but same-gender-overlap
did NOT hold steady (7/7 -> 5/7) -- a real, if smaller, regression on a
previously-perfect condition. This is a genuine mixed result, not a clean
pass or fail. Per the ground rule, this is NOT wired in without explicit
sign-off -- reported back for a decision rather than unilaterally judged
as clearing or failing the bar, since the bar's two conditions point in
different directions here.

HONEST SCOPE NOTE: tested on synthetic (TTS-constructed) overlap
mixtures only, not real recorded audio -- same limitation as today's
other speaker-verification evaluations. n=19 for the headline metric is
a small sample (one flipped clip moves it ~5 points); n=7 for the
same-gender regression is smaller still (one clip = ~14 points). Both
numbers are real and evidenced, neither should be treated as a fully
settled result at this sample size.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from udks import GENERAL_UDKS

SAMPLE_RATE = 16_000
ENROLLED_VOICE = "Microsoft Zira Desktop"
SAME_GENDER_VOICE = "Microsoft Hazel Desktop"
DIFF_GENDER_VOICE = "Microsoft David Desktop"

from wesep_extraction import WeSepExtractor  # moved to wesep_extraction.py (now a Model component); re-exported here


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Standard scale-invariant SDR (dB) -- real extraction-quality metric,
    separate from downstream verification accuracy, so a failure can be
    diagnosed as extraction-vs-verification (see module docstring)."""
    n = min(len(estimate), len(reference))
    estimate, reference = estimate[:n].astype(np.float64), reference[:n].astype(np.float64)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-10)
    proj = alpha * reference
    noise = estimate - proj
    return float(10 * np.log10((np.sum(proj**2) + 1e-10) / (np.sum(noise**2) + 1e-10)))


def main() -> None:
    print("Loading WeSep target-speaker extractor (BSRNN+ECAPA, VoxCeleb1)...")
    t0 = time.monotonic()
    extractor = WeSepExtractor()
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    print("Loading ECAPA-TDNN speaker verification model (existing project pipeline)...")
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_cache", local_strategy=LocalStrategy.COPY
    )

    def pcm_to_tensor(pcm: bytes) -> torch.Tensor:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return torch.from_numpy(audio).unsqueeze(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def synth(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            return _load_pcm_16k_mono(p)

        phrases = [u.phrase for u in GENERAL_UDKS]
        enroll_ref = synth(phrases[0], ENROLLED_VOICE, "ref.wav")

        print("\nCalibrating verification threshold: same-speaker vs different-speaker clean similarity...")
        same_speaker_calib = [synth(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [verifier.verify_batch(pcm_to_tensor(enroll_ref), pcm_to_tensor(p))[0].item() for p in same_speaker_calib]
        diff_sims = [verifier.verify_batch(pcm_to_tensor(enroll_ref), pcm_to_tensor(p))[0].item() for p in diff_speaker_calib]
        threshold = (min(same_sims) + max(diff_sims)) / 2
        print(f"  same-speaker sims: {[f'{s:.3f}' for s in same_sims]}")
        print(f"  diff-speaker sims: {[f'{s:.3f}' for s in diff_sims]}")
        print(f"  midpoint threshold: {threshold:.4f}")

        def verify(pcm: bytes) -> tuple[float, int]:
            score, _ = verifier.verify_batch(pcm_to_tensor(enroll_ref), pcm_to_tensor(pcm))
            score = float(score[0])
            return score, int(score >= threshold)

        # --- 1. Same speaker (sanity check, n=3) ---
        print(f"\n{'=' * 70}\n1. SAME SPEAKER (n=3) -- direct verify, no extraction needed, sanity check\n{'=' * 70}")
        same_test = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[6:9])]
        same_results = [verify(pcm) for pcm in same_test]
        for i, (s, p) in enumerate(same_results):
            print(f"  {i}: score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'})")
        same_correct = sum(1 for _, p in same_results if p == 1)

        # --- 2. Same-gender overlap, extract then verify (n=7) ---
        print(f"\n{'=' * 70}\n2. SAME-GENDER OVERLAP, extract-then-verify (n=7) -- want prediction=1\n{'=' * 70}")
        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_results, sg_sisdri, sg_latencies = [], [], []
        for i in range(7):
            mixed = overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")
            t0 = time.monotonic()
            extracted = extractor.extract(mixed, enroll_ref)
            lat = time.monotonic() - t0
            sg_latencies.append(lat)
            ref_audio = np.frombuffer(enrolled_clean[i], dtype="<i2").astype(np.float32)
            mix_audio = np.frombuffer(mixed, dtype="<i2").astype(np.float32)
            ext_audio = np.frombuffer(extracted, dtype="<i2").astype(np.float32)
            sdr_before = si_sdr(mix_audio, ref_audio)
            sdr_after = si_sdr(ext_audio, ref_audio)
            sg_sisdri.append(sdr_after - sdr_before)
            s, p = verify(extracted)
            sg_results.append((s, p))
            print(f"  {i}: SI-SDRi={sdr_after - sdr_before:+.2f}dB (mix={sdr_before:.2f}->extracted={sdr_after:.2f}) "
                  f"verify_score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'}) latency={lat:.2f}s")
        sg_correct = sum(1 for _, p in sg_results if p == 1)

        # --- 3. Different-gender overlap, extract then verify (n=19) ---
        print(f"\n{'=' * 70}\n3. DIFFERENT-GENDER OVERLAP, extract-then-verify (n=19) -- want prediction=1\n{'=' * 70}")
        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_results, dg_sisdri, dg_latencies = [], [], []
        for i in range(19):
            mixed = overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")
            t0 = time.monotonic()
            extracted = extractor.extract(mixed, enroll_ref)
            lat = time.monotonic() - t0
            dg_latencies.append(lat)
            ref_audio = np.frombuffer(enrolled_clean19[i], dtype="<i2").astype(np.float32)
            mix_audio = np.frombuffer(mixed, dtype="<i2").astype(np.float32)
            ext_audio = np.frombuffer(extracted, dtype="<i2").astype(np.float32)
            sdr_before = si_sdr(mix_audio, ref_audio)
            sdr_after = si_sdr(ext_audio, ref_audio)
            dg_sisdri.append(sdr_after - sdr_before)
            s, p = verify(extracted)
            dg_results.append((s, p))
            print(f"  {i:2d}: SI-SDRi={sdr_after - sdr_before:+.2f}dB (mix={sdr_before:.2f}->extracted={sdr_after:.2f}) "
                  f"verify_score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'}) latency={lat:.2f}s")
        dg_correct = sum(1 for _, p in dg_results if p == 1)

        print(f"\n{'=' * 70}\nSUMMARY (WeSep extract-then-verify, ECAPA threshold={threshold:.4f})\n{'=' * 70}")
        print(f"  Same speaker (direct, n=3):                 {same_correct}/3")
        print(f"  Same-gender overlap (extract-then-verify, n=7):     {sg_correct}/7  "
              f"(compare: direct-verify baseline was 7/7)")
        print(f"  Different-gender overlap (extract-then-verify, n=19): {dg_correct}/19 ({100*dg_correct/19:.0f}%)  "
              f"(compare: direct-verify baseline was ~21%, 4/19)")
        print(f"\n  Extraction quality (SI-SDRi, dB -- positive = extraction improved over raw mixture):")
        print(f"    same-gender overlap:      mean={statistics.mean(sg_sisdri):+.2f}dB  "
              f"min={min(sg_sisdri):+.2f}dB  max={max(sg_sisdri):+.2f}dB")
        print(f"    different-gender overlap: mean={statistics.mean(dg_sisdri):+.2f}dB  "
              f"min={min(dg_sisdri):+.2f}dB  max={max(dg_sisdri):+.2f}dB")
        print(f"\n  Latency (extraction only, CPU, per clip):")
        all_lat = sg_latencies + dg_latencies
        print(f"    n={len(all_lat)} mean={statistics.mean(all_lat):.2f}s median={statistics.median(all_lat):.2f}s "
              f"max={max(all_lat):.2f}s")
        print(f"\n  Wire-in bar: different-gender-overlap >= ~60% with same-speaker/same-gender-overlap holding steady.")

        print(f"\n{'=' * 70}\nHONEST SCOPE NOTE (mandatory)\n{'=' * 70}")
        print("  Tested on SYNTHETIC (TTS-constructed) overlap mixtures, same methodology as")
        print("  today's ECAPA-TDNN/WavLM speaker-verification tests -- NOT real recorded")
        print("  overlapping audio. n=19 for the target metric is a SMALL SAMPLE -- a single")
        print("  clip flipping changes the headline percentage by ~5 points. Treat any")
        print("  pass/fail verdict at this n as a real, evidenced signal, not a settled result.")


if __name__ == "__main__":
    main()
