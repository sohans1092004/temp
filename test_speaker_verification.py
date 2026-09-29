"""Speaker verification (NOT diarization -- see module docstring
distinction below) using speechbrain's pretrained ECAPA-TDNN
(speechbrain/spkrec-ecapa-voxceleb). Genuinely new investigation this
session, separate/additive to the frozen overlapping-speech/UDK
guardrail work.

Verification vs. diarization, stated explicitly per the task: diarization
(overlap_gate.py's pyannote pipeline, used earlier this session) answers
"how many speakers, and when did each one talk" with no prior knowledge
of who they are. Verification (this file) answers a different question:
"is THIS specific segment the one known, enrolled speaker, yes or no" --
given a reference sample of a specific person's voice in advance. They
are complementary, not interchangeable: this project's threat model (a
registered app user vs. an unknown attacker) is exactly the verification
case, not diarization's "separate two unknowns" case.

Synthetic test, not real clips: no verified ground-truth speaker
identity exists for real_recordings/mixed's clean clips (never confirmed
which files share a speaker), so a synthetic test with KNOWN ground
truth is more honest than guessing from unlabeled real audio. Windows
SAPI TTS (same 3 voices already used in evaluate_pipeline_corpus.py):
Zira and Hazel are both female (the hardest same-gender case), David is
male (the easier case).
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from audio_augment import overlap
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from udks import GENERAL_UDKS

SAMPLE_RATE = 16_000
ENROLLED_VOICE = "Microsoft Zira Desktop"  # the "registered app user"
SAME_GENDER_VOICE = "Microsoft Hazel Desktop"  # hardest case: different speaker, same gender
DIFF_GENDER_VOICE = "Microsoft David Desktop"  # easier case: different speaker, different gender


def pcm_to_tensor(pcm: bytes) -> torch.Tensor:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0)


def main() -> None:
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    print("Loading ECAPA-TDNN speaker verification model...")
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_cache", local_strategy=LocalStrategy.COPY
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("Synthesizing enrollment + test clips (3 voices x several UDK phrases)...")
        phrases = [u.phrase for u in GENERAL_UDKS[:8]]

        def synth(text: str, voice: str, name: str) -> bytes:
            wav_path = tmp_path / name
            _synthesize_wav(text, voice, wav_path)
            return _load_pcm_16k_mono(wav_path)

        enrolled_clips = [synth(p, ENROLLED_VOICE, f"enroll_{i}.wav") for i, p in enumerate(phrases[:2])]
        enrollment_ref = enrolled_clips[0]  # the one enrollment sample

        same_speaker_test_clips = [synth(p, ENROLLED_VOICE, f"same_{i}.wav") for i, p in enumerate(phrases[2:5])]
        same_gender_diff_speaker_clips = [synth(p, SAME_GENDER_VOICE, f"diffsame_{i}.wav") for i, p in enumerate(phrases[2:5])]
        diff_gender_diff_speaker_clips = [synth(p, DIFF_GENDER_VOICE, f"diffgender_{i}.wav") for i, p in enumerate(phrases[2:5])]

        def verify(test_pcm: bytes) -> tuple[float, int]:
            score, prediction = verifier.verify_batch(
                pcm_to_tensor(enrollment_ref), pcm_to_tensor(test_pcm)
            )
            return float(score[0]), int(prediction[0])

        print(f"\n{'=' * 70}\n1. SAME SPEAKER (enrolled voice, different phrases) -- want prediction=1\n{'=' * 70}")
        same_results = [verify(pcm) for pcm in same_speaker_test_clips]
        for i, (score, pred) in enumerate(same_results):
            print(f"  clip {i}: score={score:.4f} prediction={pred} ({'CORRECT' if pred == 1 else 'WRONG -- false reject'})")
        same_correct = sum(1 for _, p in same_results if p == 1)

        print(f"\n{'=' * 70}\n2. DIFFERENT SPEAKER, SAME GENDER (hardest case) -- want prediction=0\n{'=' * 70}")
        diff_same_results = [verify(pcm) for pcm in same_gender_diff_speaker_clips]
        for i, (score, pred) in enumerate(diff_same_results):
            print(f"  clip {i}: score={score:.4f} prediction={pred} ({'CORRECT' if pred == 0 else 'WRONG -- false accept'})")
        diff_same_correct = sum(1 for _, p in diff_same_results if p == 0)

        print(f"\n{'=' * 70}\n3. DIFFERENT SPEAKER, DIFFERENT GENDER (easier case) -- want prediction=0\n{'=' * 70}")
        diff_gender_results = [verify(pcm) for pcm in diff_gender_diff_speaker_clips]
        for i, (score, pred) in enumerate(diff_gender_results):
            print(f"  clip {i}: score={score:.4f} prediction={pred} ({'CORRECT' if pred == 0 else 'WRONG -- false accept'})")
        diff_gender_correct = sum(1 for _, p in diff_gender_results if p == 0)

        print(f"\n{'=' * 70}\n4. OVERLAPPING MIXTURE: enrolled voice mixed with an unknown voice\n{'=' * 70}")
        print("(verification on the MIXED audio directly, no separation -- does the enrolled voice's")
        print("presence still register, or does the second voice mask it? mode='truncate' -- see")
        print("audio_augment.overlap()'s docstring: a pooled-embedding test needs genuine full-duration")
        print("overlap, not audio_augment's default pad-to-longer, which was found this session to")
        print("silently produce a solo-speaker tail that confounds exactly this kind of test.)")
        negative_voice_clip = synth(NEGATIVES[0], SAME_GENDER_VOICE, "neg_same.wav")
        mixed_same_gender = overlap(same_speaker_test_clips[0], negative_voice_clip, mode="truncate")
        score, pred = verify(mixed_same_gender)
        print(f"  enrolled+same-gender-unknown mixed: score={score:.4f} prediction={pred} (want 1 -- enrolled voice IS present)")

        negative_voice_clip2 = synth(NEGATIVES[1], DIFF_GENDER_VOICE, "neg_diff.wav")
        mixed_diff_gender = overlap(same_speaker_test_clips[1], negative_voice_clip2, mode="truncate")
        score2, pred2 = verify(mixed_diff_gender)
        print(f"  enrolled+diff-gender-unknown mixed: score={score2:.4f} prediction={pred2} (want 1 -- enrolled voice IS present)")

        pure_unknown_mixed = overlap(negative_voice_clip, negative_voice_clip2, mode="truncate")
        score3, pred3 = verify(pure_unknown_mixed)
        print(f"  two UNKNOWN voices mixed (enrolled voice absent): score={score3:.4f} prediction={pred3} (want 0 -- enrolled voice absent)")

        print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
        print(f"  Same-speaker recall (n={len(same_results)}): {same_correct}/{len(same_results)}")
        print(f"  Different-speaker, SAME gender rejection (n={len(diff_same_results)}, hardest case): {diff_same_correct}/{len(diff_same_results)}")
        print(f"  Different-speaker, DIFFERENT gender rejection (n={len(diff_gender_results)}): {diff_gender_correct}/{len(diff_gender_results)}")
        print(f"  Overlapping mixtures (n=3): enrolled+same-gender-unknown={pred}, enrolled+diff-gender-unknown={pred2}, two-unknowns-only={pred3}")


if __name__ == "__main__":
    main()
