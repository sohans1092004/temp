"""
STT backends, behind one interface, so the rest of the pipeline never
cares which one is running (Section 9: model_version is tracked per
component precisely so this stays swappable).

FasterWhisperSTT is the real, production-intended backend (Section 17:
use an existing pretrained model, don't train one). It needs network
access to huggingface.co to fetch model weights the first time; verified
working end-to-end against real audio — see README.md.

MockSTT lets the rest of the pipeline be built, run, and tested right
now without that network access: it "transcribes" audio by returning a
transcript that was tagged onto it when the test case was built, so the
VAD/matching/decision logic all run against something real-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Transcript:
    text: str
    model_version: str
    # Per-segment confidence metadata faster-whisper already computes on
    # every transcribe() call -- previously discarded entirely. None for
    # backends that don't produce it (MockSTT, IndicFineTunedWhisperSTT)
    # so callers must treat None as "no signal," never as "confident."
    # See stt_confidence_gate.py for how these feed the decision engine.
    avg_logprob: float | None = None
    temperature: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class STTBackend(Protocol):
    def transcribe(self, pcm: bytes) -> Transcript: ...


class MockSTT:
    """Returns whatever transcript the caller attaches to a segment,
    via `pcm_to_transcript`, instead of doing real speech recognition."""

    def __init__(self):
        self._pcm_to_transcript: dict[bytes, str] = {}
        self.model_version = "mock-stt-0"

    def register(self, pcm: bytes, transcript: str) -> None:
        self._pcm_to_transcript[pcm] = transcript

    def transcribe(self, pcm: bytes) -> Transcript:
        text = self._pcm_to_transcript.get(pcm, "")
        return Transcript(text=text, model_version=self.model_version)


class FasterWhisperSTT:
    """Real backend — faster-whisper, a pretrained model, no training
    data of our own required (Section 17). Requires network access to
    huggingface.co on first run to fetch model weights; this sandbox's
    egress allowlist doesn't include that host (pypi/npm/GitHub only),
    so this class is here ready to use, but untested in this session."""

    def __init__(self, model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # deferred import

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_version = f"faster-whisper-{model_size}"

    def transcribe(self, pcm: bytes) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        segments = list(self._model.transcribe(audio, language="en")[0])
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # Aggregated across segments, same reasoning as TieredWhisperSTT's
        # existing avg_logprob averaging: mean for avg_logprob/no_speech_prob
        # (an overall confidence read), max for temperature/compression_ratio
        # (ANY segment falling back to sampling, or looping/repeating, is
        # itself the signal -- averaging it away with confident segments
        # would hide exactly the case stt_confidence_gate.py exists to catch).
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else None
        temperature = max((s.temperature for s in segments), default=None)
        compression_ratio = max((s.compression_ratio for s in segments), default=None)
        no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments) if segments else None
        return Transcript(
            text=text,
            model_version=self.model_version,
            avg_logprob=avg_logprob,
            temperature=temperature,
            compression_ratio=compression_ratio,
            no_speech_prob=no_speech_prob,
        )


PARAKEET_DIR = Path(__file__).parent / "parakeet_model"


class ParakeetSTT:
    """NVIDIA Parakeet TDT 0.6B v2 via onnx-asr (ONNX Runtime -- no NeMo, so
    none of the lightning version conflict dual_asr_guard.py's docstring
    records for nemo_toolkit). Used as a SECOND transcript alongside
    FasterWhisperSTT, alerting if EITHER alerts: the two fail differently
    (Whisper on clipped/loud audio, Parakeet on short phrases under
    overlapping speech). Measured on prepped_data x 6 audio conditions
    (2026-09-24): Whisper+Parakeet 82/84 danger clips vs Whisper 78/84,
    identical false alarms (49/96).

    Gives no decoding metadata, so avg_logprob etc. stay None, which
    stt_confidence_gate.py reads as "no signal" -- never as suspect.

    ~2.4 GB, downloaded to `model_dir` on first use (UDK_PARAKEET_DIR
    overrides the default ./parakeet_model). A plain directory on purpose:
    onnxruntime rejects the HF cache's symlinked blobs on Windows
    ("External data path escapes model directory")."""

    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu"):
        import os

        import onnx_asr
        import onnxruntime as ort

        model_dir = Path(model_dir or os.environ.get("UDK_PARAKEET_DIR", PARAKEET_DIR))
        if device == "cuda":
            try:  # onnxruntime-gpu >= 1.21: load CUDA/cuDNN from torch's pip-installed libs
                ort.preload_dlls()
            except Exception:
                pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self._model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(model_dir), providers=providers)
        # What the encoder session really got: onnxruntime silently falls back to
        # CPU when CUDA libs fail to load (e.g. a CUDA-13 onnxruntime-gpu on a
        # CUDA-12 machine), so callers can report it instead of assuming.
        try:
            self.providers = self._model.asr._encoder.get_providers()  # onnx-asr 0.12 internals
        except Exception:
            self.providers = ort.get_available_providers()
        self.model_version = "parakeet-tdt-0.6b-v2"

    def transcribe(self, pcm: bytes) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < 1600:  # < 0.1 s: nothing to transcribe
            return Transcript(text="", model_version=self.model_version)
        return Transcript(text=self._model.recognize(audio, sample_rate=16000).strip(), model_version=self.model_version)


# Real measured margin (evaluate_pipeline_corpus.py's pitch_tempo_distorted
# condition): clean baseline avg_logprob never drops below -0.458 (mean
# -0.385, n=20); the 12 real failures on that condition ranged -0.51 to
# -0.91. -0.5 sits cleanly between the two with no overlap in this corpus.
TIERED_CONFIDENCE_THRESHOLD = -0.5


class TieredWhisperSTT:
    """Confidence-gated two-tier fallback for Section 12's
    pitch_tempo_distorted gap (evaluate_pipeline_corpus.py measured 40-45%
    recall on that condition with tiny.en alone). faster-whisper already
    computes avg_logprob per segment; tiny.en just never surfaced it
    before this. Real measured result re-transcribing tiny.en's
    low-confidence segments with base.en: 7/12 of the real failures
    recovered (40% -> 75% recall on that condition) with 0 new false
    positives introduced on 40 negative clips (clean + noisy/muffled) --
    see README.md. Not free: the fallback triggered on 15/40 of those
    negatives too (noise alone also depresses tiny.en's confidence), so
    this trades real CPU latency on ~1/3 of ordinary noisy audio for a
    large recall gain on genuinely distorted audio, with no measured
    downside on FPR. Kept separate from FasterWhisperSTT (not a
    parameter on it) so existing callers that just want the fast single
    pass are unaffected."""

    def __init__(
        self,
        fast_model_size: str = "tiny.en",
        fallback_model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        confidence_threshold: float = TIERED_CONFIDENCE_THRESHOLD,
    ):
        from faster_whisper import WhisperModel  # deferred import

        self._fast = WhisperModel(fast_model_size, device=device, compute_type=compute_type)
        self._fallback = WhisperModel(fallback_model_size, device=device, compute_type=compute_type)
        self._threshold = confidence_threshold
        self.model_version = f"faster-whisper-{fast_model_size}+{fallback_model_size}-tiered"

    def transcribe(self, pcm: bytes) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        segments = list(self._fast.transcribe(audio, language="en")[0])
        text = " ".join(seg.text.strip() for seg in segments).strip()
        avg_logprob = sum(seg.avg_logprob for seg in segments) / len(segments) if segments else -999.0
        if avg_logprob >= self._threshold:
            return Transcript(text=text, model_version=self.model_version)

        fallback_segments = list(self._fallback.transcribe(audio, language="en")[0])
        fallback_text = " ".join(seg.text.strip() for seg in fallback_segments).strip()
        return Transcript(text=fallback_text, model_version=f"{self.model_version}(fallback)")


# Real, language-specific fine-tuned Whisper checkpoints (community
# fine-tunes on real Indic-language ASR data), used INSTEAD of the
# generic multilingual faster-whisper model for these four languages --
# real end-to-end audio testing found the generic model structurally
# could not produce correct Telugu script at any size tried (0% recall
# on "tiny", still only 5% on "medium" -- consistently transcribing into
# Devanagari instead of Telugu script, not a capacity problem), and was
# too garbled for Kannada even at "medium" (45%).
#
# REAL LATENCY BUG FOUND AND FIXED: the first version of this class ran
# these checkpoints via plain `transformers` WhisperForConditionalGeneration
# .generate() on CPU, because the fine-tunes only ship in that format, not
# faster-whisper's CTranslate2 format. Measured real per-segment latency:
# ~37 SECONDS for a single 2-second clip on the "medium" checkpoint --
# nowhere close to the "detect and alarm within seconds" requirement this
# entire product exists for. Root cause: plain PyTorch autoregressive
# generate() on CPU is dramatically slower than CTranslate2's optimized
# inference (the same gap that makes English's tiny.en+faster-whisper
# viable for real-time in the first place), compounded by "medium" being
# ~20x more parameters than "tiny".
#
# Fixed two ways, both real and measured, not assumed:
#  1. Convert the checkpoint to CTranslate2 format (ctranslate2 is already
#     a transitive dependency of faster-whisper) and run it through
#     faster_whisper.WhisperModel instead of raw transformers -- same
#     model weights, ~2.5x faster on "medium" alone (37s -> 14.4s).
#  2. Test smaller fine-tuned sizes instead of assuming "medium" (the
#     first choice made, picked without checking) was necessary. Real
#     result, converted + measured for every available size per language:
#
#       Language  Size     Recall   Mean latency
#       Telugu    medium   100%     14.41s (converted) / 37.29s (uncoveted)
#       Telugu    small    100%      5.97s
#       Telugu    base     100%      1.90s   <- picked: no accuracy cost at all
#       Kannada   medium    95%     (not re-measured after the fix; too slow to matter)
#       Kannada   small     90%      6.99s  (a real, small accuracy cost)
#       Kannada   base      95%      2.33s   <- picked: BEATS "small" on both axes
#       Hindi     medium    95%     (not re-measured; generic model already covered this size class)
#       Hindi     small     95%      5.99s   <- picked: no accuracy cost, no "base" exists
#       Tamil     medium   100%     (ditto)
#       Tamil     small    100%      6.81s   <- picked: no accuracy cost, no "base" exists
#
# The lesson: "medium" was never actually necessary for accuracy on this
# corpus for ANY of the four languages -- it was the first size tried,
# not a measured choice, and its unchecked latency cost would have made
# the whole multilingual feature non-viable for the real-time requirement
# this product is FOR. Real latency has to be measured for every model
# choice, the same way accuracy already is throughout this project.
INDIC_FINETUNED_MODELS: dict[str, str] = {
    "te": "vasista22/whisper-telugu-base",
    "kn": "vasista22/whisper-kannada-base",
    "hi": "vasista22/whisper-hindi-small",
    "ta": "vasista22/whisper-tamil-small",
}

# Where converted CTranslate2 models are cached after the one-time
# conversion (see IndicFineTunedWhisperSTT.__init__) -- subsequent loads
# just read from here, no re-conversion or re-download.
INDIC_CT2_CACHE_DIR = Path(__file__).parent / "indic_ct2_models"


class IndicFineTunedWhisperSTT:
    """Real backend for a single Indic language, using a fine-tune from
    INDIC_FINETUNED_MODELS (or an explicit model_name override), run via
    faster-whisper's CTranslate2 engine -- see the module-level comment
    above for the real latency numbers behind why this isn't plain
    `transformers` anymore. First construction for a given model
    downloads it and converts it to CTranslate2 format into
    INDIC_CT2_CACHE_DIR (a real, one-time cost of ~1-2 minutes); later
    constructions just load the cached conversion, same as
    FasterWhisperSTT's usual startup cost. Deferred imports, same
    reasoning as the other real backends in this file."""

    def __init__(self, language: str, model_name: str | None = None):
        import re

        from faster_whisper import WhisperModel

        model_name = model_name or INDIC_FINETUNED_MODELS[language]
        ct2_dir = INDIC_CT2_CACHE_DIR / model_name.rsplit("/", 1)[-1]
        if not (ct2_dir / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter

            ct2_dir.mkdir(parents=True, exist_ok=True)
            TransformersConverter(model_name).convert(str(ct2_dir), quantization="int8", force=True)
            # faster-whisper also needs the tokenizer/preprocessor files
            # alongside model.bin -- the converter only writes the model.
            from huggingface_hub import hf_hub_download
            import shutil

            for filename in ("tokenizer_config.json", "preprocessor_config.json", "vocabulary.json"):
                try:
                    src = hf_hub_download(model_name, filename)
                    shutil.copy(src, ct2_dir / filename)
                except Exception:
                    pass

        self._model = WhisperModel(str(ct2_dir), device="cpu", compute_type="int8")
        self._language = language
        self.model_version = f"whisper-finetuned-ct2-{model_name.rsplit('/', 1)[-1]}"
        # This checkpoint family's tokenizer doesn't register its special
        # tokens (<|startoftranscript|><|te|><|transcribe|><|notimestamps|>)
        # in a way skip_special_tokens catches under the plain-transformers
        # path this class used to take -- kept as a defensive strip here
        # too in case faster-whisper's own decoding has the same gap for
        # this checkpoint family.
        self._special_token_re = re.compile(r"<\|[^|]*\|>")

    def transcribe(self, pcm: bytes) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        # beam_size=1 (greedy) instead of faster-whisper's default 5: real
        # measurement across all four languages found zero accuracy cost
        # (each stayed at its exact same recall) for a real latency win
        # (7-26% faster depending on language) -- a free improvement once
        # measured, not assumed.
        segments, _info = self._model.transcribe(audio, language=self._language, beam_size=1)
        raw_text = " ".join(seg.text.strip() for seg in segments).strip()
        text = self._special_token_re.sub("", raw_text).strip()
        return Transcript(text=text, model_version=self.model_version)
