"""Blind 2-voice separation + per-voice transcription of the multi-speaker
clips in prepped_data/ (TEST DATA ONLY), so the quieter speaker's words --
dropped by single-stream Whisper during overlap (see
prepped_speaker_transcripts.log) -- can be checked by ear.

SepFormer (speechbrain/sepformer-wsj02mix, trained on 8 kHz audio) splits
each clip into 2 streams at 8 kHz; each stream is resampled to 16 kHz and
transcribed with faster-whisper small.en. Diagnostic only: blind separation
was rejected for DETECTION (it degrades audio, see Model-snr notes), this
just recovers readable text for verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transcribe_speakers_prepped import DATA, MULTI_SPEAKER


def main() -> None:
    import librosa
    import numpy as np
    import torch
    from faster_whisper import WhisperModel, decode_audio
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    sep = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix", savedir="sepformer_cache", local_strategy=LocalStrategy.COPY
    )
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

    def text(audio16: np.ndarray) -> str:
        segs, _ = whisper.transcribe(audio16.astype(np.float32), language="en")
        return " ".join(s.text.strip() for s in segs) or "(nothing transcribed)"

    for path in sorted(p for p in DATA.glob("*.mp3") if MULTI_SPEAKER.search(p.stem)):
        audio16 = decode_audio(str(path), sampling_rate=16000)
        audio8 = librosa.resample(audio16, orig_sr=16000, target_sr=8000)
        with torch.no_grad():
            est = sep.separate_batch(torch.from_numpy(audio8).float().unsqueeze(0))[0].numpy()  # (T, 2)
        print(f"\n{'=' * 100}\n{path.stem}")
        print(f"  mixture (as before): {text(audio16)}")
        for i in range(est.shape[-1]):
            s = est[:, i]
            s = s / (np.abs(s).max() + 1e-9) * 0.9
            s16 = librosa.resample(s, orig_sr=8000, target_sr=16000)
            rms = float(np.sqrt(np.mean(s ** 2)))
            print(f"  voice {i + 1} (rms {rms:.3f}): {text(s16)}")


if __name__ == "__main__":
    main()
