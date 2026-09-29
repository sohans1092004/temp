"""WeSep target-speaker extraction as a NO_ACTION fallback for overlapping
speech -- the enrolled user's own voice is pulled out of a mixed segment
(instead of separation.py's blind 2-way split, rejected on real audio),
then re-run through STT+KWS via separation.retry_with_separation().

Self-policing (verify_wesep_negative_enrollment.py, commit 5f8e667): the
extracted audio is compared to the enrollment sample with ECAPA-TDNN; an
extraction scoring below EXTRACTION_ACCEPT_THRESHOLD is discarded. On the
validated corpus that one threshold separated every known-good extraction
(0.46-0.74) from every known-bad one (0.27-0.35) and every enrolled-absent
negative (0/24 false accepts).

ponytail: EXTRACTION_ACCEPT_THRESHOLD is one fixed value calibrated on a
single enrolled TTS voice (Zira); per-user calibration at enrollment time
if this ever reaches real users.

EVAL ONLY: the live API never stores a user's voice sample (enrollment.py
is speaker-independent by design), so api.py does not use this. See the
setup notes (checkpoint schema, import stubs, fbank preprocessing) in
test_wesep_target_extraction.py's module docstring.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
EXTRACTION_ACCEPT_THRESHOLD = 0.4111
WESEP_MODEL_DIR = Path.home() / ".wesep" / "english"
WESEP_LEGACY_SRC = Path(__file__).parent / "wesep_legacy_src"
WESPEAKER_SRC = Path(__file__).parent / "wespeaker_src"


def _install_import_stubs() -> None:
    """Bypasses two heavy, unrelated eager-import chains (wesep's CLI +
    silero-vad, wespeaker's every-model-family import) without touching
    either cloned repo's own source files."""
    for name, path in [
        ("wesep", str(WESEP_LEGACY_SRC / "wesep")),
        ("wespeaker", str(WESPEAKER_SRC / "wespeaker")),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    if "wespeaker.models.speaker_model" not in sys.modules:
        import wespeaker.models.ecapa_tdnn as ecapa_tdnn

        stub = types.ModuleType("wespeaker.models.speaker_model")
        stub.get_speaker_model = lambda name: getattr(ecapa_tdnn, name)
        sys.modules["wespeaker.models.speaker_model"] = stub


def _pcm_to_tensor(pcm: bytes):
    import torch

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0)


class WeSepExtractor:
    model_version = "wesep-bsrnn-ecapa-vox1"

    def __init__(self, device: str = "cpu"):
        import torch
        import yaml

        _install_import_stubs()
        from wesep.models.bsrnn import BSRNN

        with open(WESEP_MODEL_DIR / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        margs = dict(cfg["model_args"]["tse_model"])
        margs["spk_model_init"] = None  # fully overwritten by the checkpoint below anyway
        self._model = BSRNN(**margs)
        ckpt = torch.load(WESEP_MODEL_DIR / "avg_model.pt", map_location="cpu", weights_only=False)
        missing, unexpected = self._model.load_state_dict(ckpt["models"][0], strict=False)
        assert not missing and not unexpected, f"weight loading mismatch: missing={missing} unexpected={unexpected}"
        self._device = device
        self._model.to(device).eval()

    def _enrollment_fbank(self, pcm: bytes):
        import torchaudio

        wave = _pcm_to_tensor(pcm) * (1 << 15)
        mat = torchaudio.compliance.kaldi.fbank(
            wave, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0,
            sample_frequency=SAMPLE_RATE, window_type="hamming", use_energy=False,
        )
        mat = mat - mat.mean(dim=0, keepdim=True)
        return mat.unsqueeze(0).to(self._device)

    def extract(self, mixture_pcm: bytes, enrollment_pcm: bytes) -> bytes:
        """mixture + enrollment sample -> extracted target-speaker audio,
        same length/format (16-bit PCM) as the input mixture."""
        import torch

        with torch.no_grad():
            out, _ = self._model(_pcm_to_tensor(mixture_pcm).to(self._device), self._enrollment_fbank(enrollment_pcm))
        return np.clip(out[0].cpu().numpy() * 32767, -32768, 32767).astype(np.int16).tobytes()


class EcapaVerifier:
    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy

        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_cache", local_strategy=LocalStrategy.COPY,
            run_opts={"device": device},
        )
        self._device = device

    def score(self, enrollment_pcm: bytes, pcm: bytes) -> float:
        s, _ = self._model.verify_batch(_pcm_to_tensor(enrollment_pcm).to(self._device), _pcm_to_tensor(pcm).to(self._device))
        return float(s[0])


class WeSepTargetSeparator:
    """separation.SeparatorBackend for one enrolled user: returns the
    extracted user voice as the only stream, or no stream at all when the
    self-check says the extraction failed / the user isn't there.
    recheck_verify: also re-check weak TRIGGER_VERIFY segments (see
    separation.recheck_verify_with_separation)."""

    recheck_verify = True

    def __init__(self, enrollment_pcm: bytes, extractor=None, verifier=None,
                 threshold: float = EXTRACTION_ACCEPT_THRESHOLD, device: str = "cpu"):
        self.enrollment_pcm = enrollment_pcm
        self._extractor = extractor or WeSepExtractor(device=device)
        self._verifier = verifier or EcapaVerifier(device=device)
        self.threshold = threshold
        self.last_score: float | None = None

    def separate(self, pcm: bytes) -> list[bytes]:
        extracted = self._extractor.extract(pcm, self.enrollment_pcm)
        self.last_score = self._verifier.score(self.enrollment_pcm, extracted)
        return [extracted] if self.last_score >= self.threshold else []
