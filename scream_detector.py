"""
Acoustic scream/distress-cue detector: mathematical signal processing,
not a trained model, deliberately. Meant as a THIRD independent
corroborating signal alongside STT text matching and KWS -- never a
sole trigger. Section 4's whole design principle is no single point of
failure; the same reasoning applies here in reverse: a high acoustic
score alone must never fire an alert either, only add confidence when
something else (text or KWS) has already produced some signal. See
udk_engine.py's corroboration wiring for how this is actually gated.

Real acoustic basis, not a guessed heuristic: screams are NOT simply
"high pitch." Arnal, Flinker, Kayser, Poeppel & Giraud (Current
Biology, 2015, "Human Screams Occupy a Privileged Niche in the Sound
Spectrum") found screams are acoustically distinguished by ROUGHNESS --
fast amplitude modulation of the sound envelope in the ~30-150 Hz
range -- far faster than normal speech's syllabic modulation
(~2-8 Hz), and that this roughness is specifically what the auditory
system uses to perceive urgency, largely independent of loudness or
pitch. This module measures that directly: the fraction of the
amplitude envelope's own modulation-spectrum power that falls in the
30-150 Hz roughness band. Combined with two weaker, secondary
correlates: pitch elevation above normal conversational range, and
short-term loudness.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000

# Real acoustic-literature range (Arnal et al. 2015) for scream
# "roughness" -- amplitude-envelope modulation rate. Not to be confused
# with pitch (F0), which is a separate, weaker feature below.
ROUGHNESS_BAND_HZ = (30.0, 150.0)

# Typical adult conversational speech F0 upper bound (commonly cited
# acoustic-phonetics baseline, covers both male and female ranges) --
# used only to detect elevation above it, not as a voice classifier.
NORMAL_F0_MAX_HZ = 255.0
SCREAM_F0_SEARCH_MAX_HZ = 1000.0  # widened vs. this project's other pyin calls (fmax=400), which target normal speech, not screams


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _roughness_score(audio: np.ndarray, sr: int) -> float:
    """Fraction of the amplitude envelope's modulation-spectrum power
    falling in the real scream "roughness" band -- see module
    docstring. 0 = a smooth envelope (typical speech/silence), up to
    1 = envelope power concentrated entirely in the roughness band."""
    if len(audio) < sr // 10:  # need >=~100ms for a meaningful envelope FFT
        return 0.0
    from scipy.signal import hilbert

    envelope = np.abs(hilbert(audio))
    envelope = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1.0 / sr)
    total_power = float(np.sum(spectrum**2))
    if total_power == 0.0:
        return 0.0
    band_mask = (freqs >= ROUGHNESS_BAND_HZ[0]) & (freqs <= ROUGHNESS_BAND_HZ[1])
    band_power = float(np.sum(spectrum[band_mask] ** 2))
    return band_power / total_power


def _pitch_elevation_score(audio: np.ndarray, sr: int) -> float:
    """0-1 score for how far the segment's voiced pitch sits above
    normal conversational range, saturating at SCREAM_F0_SEARCH_MAX_HZ.
    A real, secondary correlate of vocal strain -- see module docstring
    for why roughness, not pitch, is the primary signal here."""
    import librosa

    f0, _voiced_flag, _ = librosa.pyin(audio, fmin=60, fmax=SCREAM_F0_SEARCH_MAX_HZ, sr=sr)
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return 0.0
    peak_f0 = float(np.percentile(voiced_f0, 90))  # robust to a few octave-jump pyin errors
    if peak_f0 <= NORMAL_F0_MAX_HZ:
        return 0.0
    return min(1.0, (peak_f0 - NORMAL_F0_MAX_HZ) / (SCREAM_F0_SEARCH_MAX_HZ - NORMAL_F0_MAX_HZ))


def _energy_score(audio: np.ndarray) -> float:
    """0-1 score for short-term loudness relative to a fixed reference.
    Deliberately the weakest-weighted feature: loudness alone is the
    least specific of the three (background noise/music is loud too,
    as the real-world testing in README.md's "Real-world audio
    testing" section found the hard way for other signals)."""
    rms = float(np.sqrt(np.mean(audio**2)))
    # -20 dBFS ~ "normal speech," -3 dBFS ~ "very loud," referenced to
    # full-scale int16 -- a starting point, not independently
    # calibrated against real scream recordings; see
    # calibrate_scream_detector.py for the real composite-score
    # calibration, which is what actually sets SCREAM_SCORE_THRESHOLD.
    normal_rms = 10 ** (-20 / 20)
    loud_rms = 10 ** (-3 / 20)
    if rms <= normal_rms:
        return 0.0
    return min(1.0, (rms - normal_rms) / (loud_rms - normal_rms))


class ScreamDetector:
    """Composite acoustic distress-cue score in [0, 1], pure signal
    processing -- no trained model, no network dependency, no download.
    Weighted toward roughness (the real, literature-grounded primary
    cue) over pitch and energy (real but weaker/less specific
    correlates). Deliberately NOT a standalone trigger -- see
    udk_engine.py's corroboration wiring; a high score alone never
    fires an alert on its own."""

    model_version = "scream-dsp-v1"

    # Real measurement (calibrate_scream_detector.py) found these weights
    # need to be very different from the literature-informed starting
    # guess above: pitch elevation is what actually separates RAVDESS
    # fear/angry speech from calm speech (0.05 calm -> 0.19 neutral-read
    # -> 0.49-0.51 fear/angry), while roughness barely moved (0.015 for
    # fear/angry vs 0.054 for neutral -- LOWER, not higher) -- acted
    # emotional line-readings are not the same acoustic category as an
    # actual scream, the same class of mistake this project already made
    # once treating synthetic pitch-shift as a distress proxy. Energy was
    # actively counterproductive: real podcast negatives scored HIGHER
    # than RAVDESS fear/angry purely from different recording loudness
    # normalization between datasets, not genuine vocal effort. Roughness
    # is kept at a small weight rather than dropped, because the one
    # GENUINE real scream in this project's corpus (the Vakeel Saab clip)
    # scored the highest roughness of every group tested (0.076) -- real
    # screaming may still show it even though acted "angry" reading does
    # not; not enough real scream examples to weight it higher with
    # confidence.
    ROUGHNESS_WEIGHT = 0.15
    PITCH_WEIGHT = 0.80
    ENERGY_WEIGHT = 0.05

    def score(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        audio = _pcm_to_float(pcm)
        if len(audio) == 0:
            return 0.0
        roughness = _roughness_score(audio, sample_rate)
        pitch = _pitch_elevation_score(audio, sample_rate)
        energy = _energy_score(audio)
        return float(
            self.ROUGHNESS_WEIGHT * roughness
            + self.PITCH_WEIGHT * pitch
            + self.ENERGY_WEIGHT * energy
        )


# Real calibration (calibrate_scream_detector.py): swept against RAVDESS
# fear/angry speech at STRONG intensity (n=64, real human actors) vs.
# RAVDESS-neutral + this project's own real negative recordings (n=78
# combined) -- 0.30 gives 57.8% recall / 9.0% FPR, and correctly scores
# above threshold on the one genuine real scream in this project's
# corpus (the Vakeel Saab clip, 0.320). Recall this low would be
# unacceptable for a standalone trigger -- it is exactly why this is
# wired as a corroboration-only signal (see udk_engine.py), never a
# sole trigger: a below-average acoustic detector that only ever ADDS
# confidence to an already-partially-matched segment is still a net
# improvement, whereas the same detector firing alone would mean
# missing real distress speech more often than it catches it.
SCREAM_SCORE_THRESHOLD = 0.30


def demo() -> None:
    """Smallest runnable sanity check: a synthetically "rough" signal
    (a tone amplitude-modulated at 80Hz, inside the roughness band)
    must score higher than a plain steady tone at the same pitch/energy
    -- proves the roughness feature actually measures what it claims to,
    independent of any real audio file."""
    sr = SAMPLE_RATE
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    carrier = np.sin(2 * np.pi * 300 * t)  # 300Hz tone, same for both cases

    smooth = carrier * 0.5
    rough_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 80 * t)  # 80Hz AM -- inside the roughness band
    rough = carrier * rough_envelope

    def to_pcm(x: np.ndarray) -> bytes:
        return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

    detector = ScreamDetector()
    smooth_score = detector.score(to_pcm(smooth))
    rough_score = detector.score(to_pcm(rough))
    assert rough_score > smooth_score, (
        f"roughness feature isn't discriminating: smooth={smooth_score:.3f} rough={rough_score:.3f}"
    )
    print(f"[PASS] rough-envelope signal scores higher than smooth: {rough_score:.3f} > {smooth_score:.3f}")


if __name__ == "__main__":
    demo()
