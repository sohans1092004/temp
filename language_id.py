"""Automatic language identification from audio -- replaces trusting a
client-declared language (the client may not know it, or may be wrong)
with real detection from the audio itself.

Two real, complementary models, chosen from real measurement (see
README.md's "Automatic language routing" section for the full numbers):

  - A generic multilingual Whisper model's own built-in language
    detection (a single encoder pass, no decoding -- fast) is reliable
    even on short (2-4s) audio: 100% correct on English/Hindi/Telugu/
    Tamil in real testing, but weaker on Kannada specifically (67%,
    confused with linguistically close languages like Sinhala/Tamil).
  - SpeechBrain's VoxLingua107 (ECAPA), a model purpose-built for
    language ID rather than a byproduct of an ASR decoder -- the same
    "purpose-built beats generic" pattern already found for STT/semantic
    matching/KWS in this project -- gets ALL FOUR Indic languages
    including Kannada 100% correct in real testing. The real tradeoff:
    it needs at least ~6 seconds of audio to be reliable. Below that it
    can be badly wrong (measured: Maltese/Croatian/Latin guesses on
    1.8-4s real English clips); at 6+ seconds it was 100% correct with
    0.99+ confidence every time tested.

Used together (see api.py's `_maybe_update_language`): the fast Whisper
detector picks an initial language as soon as ANY audio is available, so
detection never blocks real-time response. VoxLingua107 then rechecks
periodically once enough audio has accumulated, catching and correcting
whatever the fast detector's Kannada weakness got wrong -- two
consecutive agreeing rechecks are required before actually switching a
journey's language mid-stream (same reasoning as udk_engine.py's
REPEAT_WINDOW_S: one signal can be noise, two agreeing signals are real).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000

# Below this many seconds of audio, VoxLinguaLanguageDetector's own real
# accuracy measurement found it unreliable -- see module docstring.
MIN_RECHECK_DURATION_S = 6.0

DEFAULT_LANGUAGE = "en"

# Both detectors' native codes already match udks.py's SUPPORTED_LANGUAGES
# for all 5 languages this project supports; a detected code outside that
# set (a genuinely unsupported language) falls back to English rather
# than crashing or guessing further.
_SUPPORTED = {"en", "hi", "te", "kn", "ta"}


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class WhisperLanguageDetector:
    """Fast initial detection. Real backend, deferred import -- same
    reasoning as stt.py/kws.py: torch/faster-whisper are heavy
    dependencies a caller who doesn't need this shouldn't have to pay
    startup cost for."""

    def __init__(self, model_size: str = "small"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model_version = f"faster-whisper-{model_size}-lid"

    def detect(self, pcm: bytes) -> tuple[str, float]:
        """Returns (language, probability). An out-of-support-set
        detection falls back to DEFAULT_LANGUAGE rather than being
        passed through -- this project has no backend for it anyway."""
        audio = _pcm_to_float(pcm)
        detected, prob, _all_probs = self._model.detect_language(audio=audio)
        return (detected if detected in _SUPPORTED else DEFAULT_LANGUAGE), prob


class VoxLinguaLanguageDetector:
    """Accurate periodic recheck. Callers MUST NOT use this on audio
    shorter than MIN_RECHECK_DURATION_S -- see module docstring for the
    real measured unreliability below that duration."""

    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="voxlingua_cache",
            local_strategy=LocalStrategy.COPY,
        )
        self.model_version = "voxlingua107-ecapa-lid"

    def detect(self, pcm: bytes) -> tuple[str, float]:
        import torch

        audio = _pcm_to_float(pcm)
        signal = torch.from_numpy(audio).unsqueeze(0)
        prediction = self._model.classify_batch(signal)
        label = prediction[3][0]  # e.g. 'kn: Kannada'
        score = prediction[1].exp().item()
        detected = label.split(":")[0].strip()
        return (detected if detected in _SUPPORTED else DEFAULT_LANGUAGE), score
