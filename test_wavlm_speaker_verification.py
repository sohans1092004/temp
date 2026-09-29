"""WavLM-based speaker verification (microsoft/wavlm-base-plus-sv,
transformers.WavLMForXVector) vs. ECAPA-TDNN, on the exact same test
suite as test_speaker_verification.py -- same-speaker, same-gender
overlap, different-gender overlap -- for a direct, controlled
comparison. Uses audio_augment.overlap(mode="truncate") (the fix from
this session's confound audit), not the original padded construction.

VERDICT: DEAD END, does not clear the wire-in bar (diff-gender-overlap
>=~60% with the other two staying strong). Real result, threshold
calibrated from data (0.7809, midpoint of same-speaker vs. diff-speaker
clean similarity), not assumed:
  same speaker:                  3/3   (matches ECAPA-TDNN)
  different speaker, same gender: 3/3   (matches ECAPA-TDNN)
  same-gender overlap:           3/7 (43%)  -- WORSE than ECAPA-TDNN's 7/7 (100%)
  different-gender overlap:      6/19 (32%) -- better than ECAPA's ~21%, but far
                                                below the 60% bar
Clean single-speaker performance is comparable to ECAPA-TDNN, but WavLM
is a net regression on same-gender overlap (a case ECAPA had solved
cleanly) and still nowhere near usable on different-gender overlap.
Not wired in. Left as a standalone script, not a corroboration signal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from udks import GENERAL_UDKS

SAMPLE_RATE = 16_000
ENROLLED_VOICE = "Microsoft Zira Desktop"
SAME_GENDER_VOICE = "Microsoft Hazel Desktop"
DIFF_GENDER_VOICE = "Microsoft David Desktop"


class WavLMVerifier:
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self._model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        self._model.eval()

    def embed(self, pcm: bytes) -> torch.Tensor:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        inputs = self._extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs)
        return out.embeddings[0]

    def similarity(self, pcm_a: bytes, pcm_b: bytes) -> float:
        ea, eb = self.embed(pcm_a), self.embed(pcm_b)
        return float(torch.nn.functional.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0))[0])


def main() -> None:
    print("Loading WavLM X-Vector model...")
    verifier = WavLMVerifier()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def synth(text: str, voice: str, name: str) -> bytes:
            p = tmp_path / name
            _synthesize_wav(text, voice, p)
            return _load_pcm_16k_mono(p)

        phrases = [u.phrase for u in GENERAL_UDKS]
        enroll_ref = synth(phrases[0], ENROLLED_VOICE, "ref.wav")

        # --- Calibrate a threshold from data, not assumed ---
        print("\nCalibrating threshold: same-speaker vs. different-speaker clean similarity...")
        same_speaker_calib = [synth(p, ENROLLED_VOICE, f"cal_same_{i}.wav") for i, p in enumerate(phrases[1:6])]
        diff_speaker_calib = [synth(p, SAME_GENDER_VOICE, f"cal_diff_{i}.wav") for i, p in enumerate(phrases[1:6])]
        same_sims = [verifier.similarity(enroll_ref, pcm) for pcm in same_speaker_calib]
        diff_sims = [verifier.similarity(enroll_ref, pcm) for pcm in diff_speaker_calib]
        print(f"  same-speaker sims: {[f'{s:.3f}' for s in same_sims]}")
        print(f"  diff-speaker sims: {[f'{s:.3f}' for s in diff_sims]}")
        threshold = (min(same_sims) + max(diff_sims)) / 2
        print(f"  midpoint threshold picked from this data: {threshold:.4f}")

        def verify(pcm_a: bytes, pcm_b: bytes) -> tuple[float, int]:
            s = verifier.similarity(pcm_a, pcm_b)
            return s, int(s >= threshold)

        # --- 1. Same speaker ---
        print(f"\n{'=' * 70}\n1. SAME SPEAKER (n=3) -- want prediction=1\n{'=' * 70}")
        same_test = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[6:9])]
        same_results = [verify(enroll_ref, pcm) for pcm in same_test]
        for i, (s, p) in enumerate(same_results):
            print(f"  {i}: score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG'})")
        same_correct = sum(1 for _, p in same_results if p == 1)

        # --- 2. Different speaker, same gender ---
        print(f"\n{'=' * 70}\n2. DIFFERENT SPEAKER, SAME GENDER (n=3, hardest clean case) -- want prediction=0\n{'=' * 70}")
        diff_same_test = [synth(p, SAME_GENDER_VOICE, f"dsame_{i}.wav") for i, p in enumerate(phrases[6:9])]
        diff_same_results = [verify(enroll_ref, pcm) for pcm in diff_same_test]
        for i, (s, p) in enumerate(diff_same_results):
            print(f"  {i}: score={s:.4f} pred={p} ({'CORRECT' if p == 0 else 'WRONG'})")
        diff_same_correct = sum(1 for _, p in diff_same_results if p == 0)

        # --- 3. Same-gender overlap (n=7, matching earlier ECAPA test) ---
        print(f"\n{'=' * 70}\n3. SAME-GENDER OVERLAP (n=7, fair truncate mode) -- want prediction=1\n{'=' * 70}")
        enrolled_clean = [synth(p, ENROLLED_VOICE, f"e{i}.wav") for i, p in enumerate(phrases[:7])]
        same_gender_unknown = [synth(NEGATIVES[i], SAME_GENDER_VOICE, f"sg{i}.wav") for i in range(7)]
        sg_results = []
        for i in range(7):
            mixed = overlap(enrolled_clean[i], same_gender_unknown[i], mode="truncate")
            s, p = verify(enroll_ref, mixed)
            sg_results.append((s, p))
            print(f"  {i}: score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG false-reject'})")
        sg_correct = sum(1 for _, p in sg_results if p == 1)

        # --- 4. Different-gender overlap (n=19, matching earlier ECAPA test) ---
        print(f"\n{'=' * 70}\n4. DIFFERENT-GENDER OVERLAP (n=19, fair truncate mode) -- want prediction=1\n{'=' * 70}")
        enrolled_clean19 = [synth(p, ENROLLED_VOICE, f"e19_{i}.wav") for i, p in enumerate(phrases[1:20])]
        diff_gender_unknown19 = [synth(NEGATIVES[i % len(NEGATIVES)], DIFF_GENDER_VOICE, f"dg19_{i}.wav") for i in range(19)]
        dg_results = []
        for i in range(19):
            mixed = overlap(enrolled_clean19[i], diff_gender_unknown19[i], mode="truncate")
            s, p = verify(enroll_ref, mixed)
            dg_results.append((s, p))
            print(f"  {i:2d}: score={s:.4f} pred={p} ({'CORRECT' if p == 1 else 'WRONG false-reject'})")
        dg_correct = sum(1 for _, p in dg_results if p == 1)

        print(f"\n{'=' * 70}\nSUMMARY (WavLM, threshold={threshold:.4f})\n{'=' * 70}")
        print(f"  Same speaker:                  {same_correct}/3")
        print(f"  Different speaker, same gender: {diff_same_correct}/3")
        print(f"  Same-gender overlap:           {sg_correct}/7")
        print(f"  Different-gender overlap:      {dg_correct}/19 ({100*dg_correct/19:.0f}%)")
        print(f"\n  Compare to ECAPA-TDNN (today): same-speaker 3/3, same-gender-overlap 7/7, diff-gender-overlap ~21% (4/19)")
        print(f"  Wire-in bar: diff-gender-overlap >= ~60% with the other two staying strong.")


if __name__ == "__main__":
    main()
