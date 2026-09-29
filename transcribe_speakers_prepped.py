"""Per-speaker transcripts for the multi-speaker clips in prepped_data/
(TEST DATA ONLY -- nothing here trains or tunes on it), for manual
verification of what each person said.

  who spoke when : pyannote/speaker-diarization-community-1 (same pipeline
                   Model-snr/overlap_gate.py already uses; waveform passed
                   in-memory, see that module's torchcodec note)
  what was said  : faster-whisper small.en with word timestamps
  alignment      : each word goes to the speaker(s) active at its midpoint;
                   a word spoken while 2+ speakers are active is labelled
                   OVERLAP. Diarization can't unmix overlapped audio, so an
                   OVERLAP span is Whisper's best reading of the mixture.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(__file__).parent / "prepped_data"
MULTI_SPEAKER = re.compile(r"female.*female|\bmale\b|2 females|females|overlap", re.IGNORECASE)


def main() -> None:
    import torch
    from faster_whisper import WhisperModel, decode_audio
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline

    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN") or get_token()
    )
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

    for path in sorted(p for p in DATA.glob("*.mp3") if MULTI_SPEAKER.search(p.stem)):
        audio = decode_audio(str(path), sampling_rate=16000)
        diar = diarizer({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}).speaker_diarization
        turns = [(seg.start, seg.end, spk) for seg, _, spk in diar.itertracks(yield_label=True)]
        speakers = sorted({s for _, _, s in turns})
        overlap_s = diar.get_overlap().duration()

        segments, _ = whisper.transcribe(audio, language="en", word_timestamps=True)
        words = [w for s in segments for w in (s.words or [])]

        def who(t: float) -> str:
            active = sorted({s for a, b, s in turns if a <= t <= b})
            if len(active) > 1:
                return "OVERLAP(" + "+".join(active) + ")"
            if active:
                return active[0]
            return min(turns, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))[2] if turns else "?"

        lines, cur, buf, start = [], None, [], 0.0
        for w in words:
            label = who((w.start + w.end) / 2)
            if label != cur and buf:
                lines.append((start, cur, "".join(buf).strip()))
                buf = []
            if not buf:
                start = w.start
            cur = label
            buf.append(w.word)
        if buf:
            lines.append((start, cur, "".join(buf).strip()))

        print(f"\n{'=' * 100}\n{path.stem}")
        print(f"  speakers detected: {len(speakers)} {speakers}   overlapped speech: {overlap_s:.1f}s of {len(audio) / 16000:.1f}s")
        print("  who spoke when:  " + "  ".join(f"[{a:.1f}-{b:.1f}s {s}]" for a, b, s in turns))
        print("  full transcript: " + " ".join(w.word.strip() for w in words))
        print("  by speaker:")
        for t, label, text in lines:
            print(f"    {t:5.1f}s  {label:<28} {text}")


if __name__ == "__main__":
    main()
