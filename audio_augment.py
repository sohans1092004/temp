"""
Signal-level augmentation for building a Section 12-style test corpus
without real distress recordings -- the same technique Section 17
already recommends for bootstrapping narrow-vocabulary KWS training data
("TTS-synthesized recordings... layered with noise/muffling augmentation
... and pitch/pace variation informed by emotional-speech datasets to
approximate distress delivery"), applied here to build a broader
whole-pipeline evaluation corpus (build_section12_corpus.py), not just
KWS reference clips.

Every function takes/returns 16-bit PCM bytes at vad.py's SAMPLE_RATE,
matching this project's audio convention everywhere else.
"""

from __future__ import annotations

import numpy as np


def _to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _to_pcm(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def pitch_shift(pcm: bytes, sample_rate: int, n_steps: float) -> bytes:
    """NOT a validated distress proxy -- see validate_distress_with_tess.py.
    Real fear/angry speech (TESS, real human recordings) shifts pitch up
    by ~+5.6 to +8.1 semitones with only ~3-4 points of STT accuracy
    loss vs. neutral. The SAME magnitude of shift applied digitally here
    causes catastrophic STT degradation (evaluate_pipeline_corpus.py's
    original "distress_sim" condition, ~40-45% recall vs. 100% baseline).
    The gap isn't the shift amount, it's the technique: a real vocal
    tract at a higher pitch still produces clean, coherent harmonics; a
    phase-vocoder pitch-shift smears them into artifacts a real
    higher-pitched voice never has. Kept for what it's actually useful
    for -- a generic audio-distortion robustness check -- not as a
    distress-register simulation. See README.md for the full comparison."""
    import librosa

    audio = librosa.effects.pitch_shift(_to_float(pcm), sr=sample_rate, n_steps=n_steps)
    return _to_pcm(audio)


def time_stretch(pcm: bytes, rate: float) -> bytes:
    """Same caveat as pitch_shift: real fear/angry speech (TESS) IS
    faster than neutral (~1.28-1.31x, measured) -- the tempo change
    alone isn't the problem, it's combining it with pitch_shift's
    phase-vocoder artifacts. rate > 1.0 = faster."""
    import librosa

    audio = librosa.effects.time_stretch(_to_float(pcm), rate=rate)
    return _to_pcm(audio)


def muffle(pcm: bytes, sample_rate: int, cutoff_hz: float = 1500.0, attenuation_db: float = 6.0) -> bytes:
    """Pocket/bag simulation (Section 2): fabric muffling heavily
    attenuates high-frequency content and reduces overall volume --
    physical facts, not a model-quality gap, per Section 2's own framing."""
    from scipy.signal import butter, filtfilt

    audio = _to_float(pcm)
    b, a = butter(4, cutoff_hz / (sample_rate / 2), btype="low")
    filtered = filtfilt(b, a, audio)
    attenuated = filtered * (10 ** (-attenuation_db / 20))
    return _to_pcm(attenuated)


def _synthesize_noise(n_samples: int, sample_rate: int, kind: str) -> np.ndarray:
    """Procedural stand-in for real environmental noise recordings
    (traffic/crowd), since none are available here -- band-limited
    filtered white noise, not a claim of acoustic realism."""
    from scipy.signal import butter, filtfilt

    rng = np.random.default_rng(seed=hash(kind) % (2**32))
    white = rng.normal(0, 1, n_samples).astype(np.float32)
    # "traffic": low-rumble dominant; "crowd": broader mid-band chatter-ish energy
    low, high = (50, 400) if kind == "traffic" else (200, 2000)
    b, a = butter(4, [low / (sample_rate / 2), high / (sample_rate / 2)], btype="band")
    noise = filtfilt(b, a, white)
    return noise / (np.abs(noise).max() + 1e-9)


def add_noise(pcm: bytes, sample_rate: int, snr_db: float, kind: str = "traffic") -> bytes:
    """Mixes in background noise at a target signal-to-noise ratio."""
    audio = _to_float(pcm)
    noise = _synthesize_noise(len(audio), sample_rate, kind)
    signal_power = np.mean(audio**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    return _to_pcm(audio + noise * scale)


def overlap(pcm_a: bytes, pcm_b: bytes, mode: str = "pad") -> bytes:
    """Two speakers at once (Section 2: the hardest case for VAD/STT,
    and exactly the scenario -- an altercation, a struggle -- this
    product cares most about).

    mode="pad" (default, and this project's established convention --
    every existing numbers built against this function, e.g.
    separation.py's 3/10 recovery and evaluate_pipeline_corpus.py's
    "overlapping" condition, use this): the shorter clip is padded with
    silence, so the longer clip continues alone past where the shorter
    one ends -- not unrealistic (real overlapping speech isn't
    perfectly time-aligned either), but it means a caller that pools a
    SINGLE score/embedding over the whole clip (a speaker-verification
    embedding, a clip-level classifier score) is partly scoring solo
    audio, not genuine overlap, whichever input happens to be longer.

    mode="truncate" (added after a real bug found this session -- a
    speaker-verification test's "different-gender overlap" accuracy
    number was initially confounded by exactly this: cases where the
    enrolled speaker's clip was shorter than the interferer's showed a
    solo-interferer-free tail of pure enrolled speech, inflating
    apparent accuracy for reasons having nothing to do with real
    overlap): both clips truncated to the shorter one's length, so the
    mixed audio is genuinely overlapping for its entire duration. Use
    this whenever a single score/embedding will be pooled over the
    whole result; mode="pad" remains correct for segment-level
    processing (VAD will naturally split the solo tail into its own
    segment) and for comparability with this project's existing
    "pad"-based numbers."""
    a, b = _to_float(pcm_a), _to_float(pcm_b)
    if mode == "truncate":
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        return _to_pcm((a + b) * 0.5)
    if mode != "pad":
        raise ValueError(f"overlap(): unknown mode {mode!r}, expected 'pad' or 'truncate'")
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a)))
    b = np.pad(b, (0, n - len(b)))
    mixed = (a + b) * 0.5  # avoid clipping when both are near full-scale
    return _to_pcm(mixed)
