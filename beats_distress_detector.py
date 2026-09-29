"""Trained-model acoustic distress-cue detector, using Microsoft's BEATs
(https://arxiv.org/abs/2212.09058), fine-tuned checkpoint on full AudioSet
(527 classes). Same role and same interface contract as
scream_detector.ScreamDetector (a 0-1 corroborating score, never a
standalone trigger -- see that module's docstring and udk_engine.py's
scream_score wiring for the corroboration design this slots into): this
supersedes that module's pure-DSP roughness/pitch/energy heuristic after
real measurement (see DISTRESS_SCORE_THRESHOLD's comment below) found it a
clear improvement. scream_detector.py is left in place, standalone and
still independently testable, just no longer wired into api.py.

Checkpoint gotcha found and fixed building this: BEATs' own repo ships
several similarly-named checkpoints, and the most obvious-looking one
(BEATs_iter3_plus_AS2M.pt, the file Microsoft's README's own example
snippet points at) is actually the SELF-SUPERVISED PRETRAINED BACKBONE
ONLY -- no classifier head, no label_dict, calling extract_features() on
it returns raw embeddings, not class probabilities. The one that actually
has a trained AudioSet classifier (cfg['finetuned_model']=True, real
predictor.weight/bias, a 527-entry label_dict) is the differently-named
BEATs_iter3_finetuned_on_AS2M_cpt1.pt. Verified by inspecting both
checkpoints' state_dict keys directly rather than trusting the filename.

Checkpoint downloaded on first use via huggingface_hub (already installed
-- transitive dep of transformers/datasets, same as every other opt-in
heavy backend here) from https://huggingface.co/lpepino/beats_ckpts, a
mirror of Microsoft's own release (MIT) -- Microsoft's own README links
are OneDrive share links that require an interactive browser session, not
scriptable. Cached in beats_cache/, same pattern as voxlingua_cache/
(language_id.py). BEATs.py/backbone.py/modules.py vendored from
https://github.com/microsoft/unilm/tree/master/beats (MIT, see
beats_vendor/LICENSE) -- pure torch/torchaudio/numpy, no fairseq
dependency despite that module's own header comment.

Curated distress subset: resolved by loading the checkpoint's label_dict
(index -> AudioSet mid) and cross-referencing Google's official
class_labels_indices.csv (mid -> display_name), not guessed -- indices
below are specific to this exact checkpoint file. AudioSet has no
dedicated "struggle"/"scuffle" class; Smash/crash and Slap/smack are the
closest real substitutes, scored separately via physical_score() rather
than silently folded into the main corroboration score.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "beats_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

SAMPLE_RATE = 16_000
CHECKPOINT_REPO = "lpepino/beats_ckpts"
CHECKPOINT_FILENAME = "BEATs_iter3_finetuned_on_AS2M_cpt1.pt"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "beats_cache")

# idx -> AudioSet display_name, resolved from this checkpoint's label_dict
# + class_labels_indices.csv (see module docstring). Vocal distress cues:
VOCAL_DISTRESS_INDICES = {
    34: "Screaming",
    50: "Yell",
    344: "Shout",
    433: "Children shouting",
    334: "Battle cry",
    498: "Crying, sobbing",
    395: "Baby cry, infant cry",
    81: "Whimper",
    410: "Groan",
}
# Physical-altercation-adjacent cues (glass breaking, impact sounds) --
# real AudioSet classes, but weaker/less specific corroboration than a
# vocal distress cue, so scored and reported separately.
PHYSICAL_DISTRESS_INDICES = {
    399: "Glass",
    210: "Shatter",
    111: "Smash, crash",
    23: "Slap, smack",
}

# Real calibration (this project's own real audio, same methodology as
# calibrate_scream_detector.py -- real positive + real negatives, not
# synthetic): the one genuine real scream in this project's corpus (the
# Vakeel Saab clip, real_recordings/positives/) peaks at 0.454 vocal score
# in its actual scream window (45.0-47.0s of a 50.5s clip) -- already
# ahead of scream_detector.ScreamDetector's own calibrated 0.320 on the
# same clip. IMPORTANT gotcha: scoring the WHOLE 50.5s clip at once
# (mean-pooling logits over the entire duration before sigmoid, per
# BEATs.py's forward()) dilutes this down to 0.065 -- score() must be
# called per short (~1-3s) segment, e.g. per VAD segment as api.py already
# does for scream_score, never on a whole raw file. 6 real negative clips
# (ordinary podcast/talk speech, one an actual movie fight scene) screened
# the same way (2s windows, 1s hop): max score across all of them was
# 0.018, zero windows over 0.30 -- a much wider real separation margin
# than the DSP detector's 9% FPR.
DISTRESS_SCORE_THRESHOLD = 0.30

# THRESHOLD RECALIBRATION -- open, evidenced, not yet closed (see
# ACOUSTIC_VERIFY discussion, an escalation path proposed but ruled out
# elsewhere in this project's history for lack of acoustic signal on the
# synthetic overlapping-speech test corpus). That "no signal" finding was
# corpus-specific, not general: re-tested against RAVDESS fear/angry
# (strong intensity, n=64 real actors) instead of synthetic TTS, and
# 0.30 is badly miscalibrated for anything short of a full scream --
#   RAVDESS fear+angry: mean=0.052-0.084, max=0.188-0.232 (0/64 clear 0.30)
#   vs. real-negative ceiling (163 real segments, 23 files, incl. one
#   movie fight scene): max=0.0138
# Ruled out a dataset-artifact explanation first: RAVDESS-calm/happy
# (same actors/mic/recording chain) score 0.0036-0.0074 mean, right at
# the negative baseline -- the fear/angry elevation is a real,
# emotion-specific signal, not a "different dataset" effect.
# Threshold sweep (RAVDESS fear+angry recall vs. the 163-segment real
# negative set, 0% FPR measured at every level down to 0.05):
#   0.30->0% recall  0.20->4.7%  0.15->10.9%  0.10->26.6%  0.05->48.4%
# NOT YET RESOLVED, and this is the one thing standing between "real
# finding" and "safe to ship a lower threshold": the negative set above
# has no dedicated "heated argument, safe/non-dangerous" class (couples
# bickering, siblings fighting, heated debate) -- the single most
# adversarial category for this exact signal, since real anger without
# real danger would share BEATs' acoustic fingerprint for "angry"
# (raised pitch/energy/vocal strain) while being exactly the false
# positive a deployed trigger can't afford. This session's own forced-
# scoring investigation found its worst failure (95% adversarial FPR)
# exactly this way -- clean until the specific adversarial class was
# tested. Attempted to source real audio for this class via Hugging Face
# (IEMOCAP mirror: pre-extracted .pkl features, not raw audio;
# CLAPv2/MSP_podcast and hywk126/MSP_podcast: spectrogram PNGs with
# arousal/valence regression labels, not raw audio either) -- no
# freely-obtainable raw-audio source found; canonical IEMOCAP/MSP-Podcast
# require a license/application process, not an instant download.
# STATUS: real, evidenced candidate for recalibrating this threshold
# (or for reviving ACOUSTIC_VERIFY specifically), CONTINGENT on sourcing
# that one negative class and re-running this same sweep against it.
# Not validated to ship at a lower value; also not still "no signal."


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class BEATsDistressDetector:
    """Composite acoustic distress-cue score in [0, 1] from a real trained
    AudioSet classifier (BEATs), vs. scream_detector.ScreamDetector's pure
    signal-processing heuristic. Same corroboration-only role: a high
    score here must still never fire an alert alone."""

    model_version = "beats-iter3-as2m-cpt1"

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        import torch

        from BEATs import BEATs, BEATsConfig

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILENAME, cache_dir=CACHE_DIR
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        self._model = BEATs(cfg)
        self._model.load_state_dict(checkpoint["model"])
        self._device = device
        self._model.to(device).eval()
        self._torch = torch

    def label_probs(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Raw 527-class sigmoid probabilities (AudioSet is multi-label,
        not softmax) -- exposed separately from score() so callers/tests
        can inspect individual classes, not just the composite score."""
        audio = _pcm_to_float(pcm)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        if len(audio) < SAMPLE_RATE // 5:  # BEATs 16x16 patch needs >=16 fbank frames (~170ms); 100ms crashed conv2d on a pocket clip
            return np.zeros(527, dtype=np.float32)
        torch = self._torch
        source = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        padding_mask = torch.zeros_like(source, dtype=torch.bool)
        with torch.no_grad():
            probs, _ = self._model.extract_features(source, padding_mask=padding_mask)
        return probs.squeeze(0).cpu().numpy()

    def score(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        """Max probability across the curated vocal-distress classes --
        an OR, not a sum: AudioSet's multi-label sigmoid outputs are
        independent per-class probabilities, and any one class being
        confidently present (a scream OR crying OR a shout) is itself
        the corroborating signal, not their combined magnitude."""
        probs = self.label_probs(pcm, sample_rate)
        return float(max(probs[idx] for idx in VOCAL_DISTRESS_INDICES))

    def physical_score(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
        """Same idea, over the physical-altercation-adjacent classes.
        Kept separate from score() -- see module docstring. Not currently
        wired into udk_engine.py's corroboration gate (only score()'s
        vocal-distress signal is), a real next step if physical-impact
        corroboration is wanted too."""
        probs = self.label_probs(pcm, sample_rate)
        return float(max(probs[idx] for idx in PHYSICAL_DISTRESS_INDICES))


def demo() -> None:
    """Smallest runnable sanity check: the model must load (downloading
    the checkpoint on first run), run without crashing, and score silence
    low. Distinguishing real screaming from real calm speech needs real
    audio (see this module's DISTRESS_SCORE_THRESHOLD comment for that
    real validation) -- a synthetic tone proves nothing about a trained
    classifier beyond "does inference run", unlike scream_detector.py's
    hand-derived acoustic formula."""
    sr = SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    pcm = (np.clip(silence, -1, 1) * 32767).astype("<i2").tobytes()

    detector = BEATsDistressDetector()
    score = detector.score(pcm)
    physical = detector.physical_score(pcm)
    assert 0.0 <= score <= 1.0, f"score out of range: {score}"
    assert 0.0 <= physical <= 1.0, f"physical_score out of range: {physical}"
    assert score < 0.3, f"1s of digital silence scored suspiciously high as distress: {score:.3f}"
    print(f"[PASS] model loads, runs, and scores silence low: vocal={score:.3f} physical={physical:.3f}")


if __name__ == "__main__":
    demo()
