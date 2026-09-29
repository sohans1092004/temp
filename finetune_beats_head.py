"""Fine-tune BEATs' classifier head on RAVDESS, properly validated -- a
standalone experiment, NOT wired into beats_distress_detector.py's
shipped path. Run this script to reproduce the full before/after report;
nothing here changes production behavior by itself.

VERDICT: DO NOT WIRE IN. Real, decisive regression on the exact axis
that matters most, despite a large win on the axis it was trained for.
Real numbers (24-actor RAVDESS split: 18 train / 6 held-out actors,
165 real_recordings/negatives segments split by file 18/5 files):

  Held-out actors (19-24, never seen in training) -- OLD (AudioSet) vs NEW (fine-tuned):
    fear   (n=24): mean 0.052 -> 0.890   angry  (n=24): mean 0.050 -> 0.979
    calm   (n=24): mean 0.003 -> 0.000   happy  (n=24): mean 0.007 -> 0.140 (max 0.934)
    held-out real negatives (n=15): mean 0.004 -> 0.072 (max 0.943)

  THE DECISIVE RESULT: the one genuine real scream in this project's
  entire corpus (correctly-windowed, the same clip beats_distress_
  detector.py's own calibration is built on) scores OLD=0.4541 ->
  NEW=0.0000. The fine-tune, by specializing hard on RAVDESS's acted
  fear/angry acoustic signature, LOST the signal the original 527-class
  AudioSet head still carried for genuine screaming. This is the same
  "acted emotion is not the same acoustic category as real distress"
  lesson this project has already hit in scream_detector.py's own
  calibration notes -- fine-tuning on RAVDESS doesn't sidestep it, it
  makes it worse by concentrating the model's whole decision boundary
  on the acted signature specifically.

  Also a real, non-trivial regression on real-world negatives: the old
  head's real-negative ceiling was 0.014-0.018 across this entire
  project's history; the new head hits max=0.943 on at least one held-
  out real negative segment and max=0.934 on a held-out RAVDESS-happy
  clip -- high-arousal ordinary/happy speech now gets conflated with
  distress, the identical failure mode that already killed the SER
  arousal-scoring technique this session. A threshold strict enough to
  restore 0% FPR on held-out real negatives (0.95) drops held-out
  RAVDESS recall from 93.8% to 81.2%.

  Overfitting check: mean train-vs-held-out gap on distress clips is
  small (+0.0593, train=0.9937/held-out=0.9345) -- but the MINIMUM
  held-out distress score is 0.2297 vs. a training-set minimum of
  0.8120, a real generalization gap the mean alone hides: at least one
  held-out actor's fear/angry delivery is barely above chance under the
  new head, even though the average looks fine.

  Per the mandatory scope statement: this narrows the acted-distress
  ceiling and nothing more -- and here, narrowing that ceiling came at
  the direct cost of the one real (non-acted) distress signal available.
  Original AudioSet zero-shot head remains in place, unchanged, as the
  shipped default. beats_finetuned_head.pkl is kept as a documented,
  evidenced, standalone artifact -- not loaded by any production path.
"""

WHY: beats_distress_detector.py uses BEATs' original AudioSet-trained
527-class head zero-shot. Today's testing found "Screaming"/"Shout"
barely respond to a genuine scream while "Groan" (not an intuitive
distress class) carries real signal, and the composite 0.30 threshold
misses RAVDESS fear/angry entirely (see beats_distress_detector.py's own
THRESHOLD RECALIBRATION section). This fine-tunes a NEW, small binary
head specifically for the distress/calm boundary, keeping the pretrained
backbone frozen.

ARCHITECTURE CHANGE, exact and minimal:
  - FROZEN: everything up through BEATs' transformer encoder (patch
    embedding, positional/relative encoding, all encoder layers, the
    post-encoder layer_norm/post_extract_proj if present). No gradient
    ever touches these -- we only ever run them under torch.no_grad()
    to extract a fixed per-clip embedding.
  - REPLACED: the original `predictor` (nn.Linear(encoder_embed_dim,
    527) -> per-frame logits -> mean-pooled over time, i.e. AudioSet's
    527-way multi-label head) is not used at all for the new head. In
    its place: mean-pool the frozen encoder's per-frame hidden states
    over time (768-dim for this checkpoint), then a single
    LogisticRegression (scikit-learn -- mathematically identical to one
    nn.Linear(768, 1) + sigmoid, trained via LBFGS instead of a manual
    SGD loop; this is "a small linear layer being retrained", just using
    a well-tested convex optimizer instead of writing one). Note mean-
    pooling commutes with a single linear layer (mean(Linear(x)) ==
    Linear(mean(x))), so pooling before vs. after the head is
    mathematically equivalent to the original architecture's own
    pool-after-linear order -- this is not a shortcut that changes what
    is being fit.

DATA SPLIT -- actor-disjoint, not clip-disjoint (this is the part the
task explicitly warned matters): RAVDESS has 24 actors. This project's
existing collect_ravdess() helper (calibrate_scream_detector.py) only
ever downloads actors 1-8 -- every "n=32" RAVDESS number quoted
elsewhere in this project's docs/code comes from that 8-actor subset,
NOT the full 24. Verified all 24 are actually downloadable from the same
HF mirror before relying on this. TRAIN_ACTORS = 1-18, HELDOUT_ACTORS =
19-24 (a fixed, non-cherry-picked split -- last 6 by number -- RAVDESS's
odd/even actor numbering alternates gender, so both splits get both
genders). Fear+angry (intensity=strong) = distress/positive. Calm+happy
(intensity=normal) = control/negative, same categories and intensities
this project already uses elsewhere for comparability.

real_recordings/negatives (163 segments total across all files, same
VAD-segmentation method as this session's SER false-positive check) is
also split BY FILE (not by segment) into a train fold and a held-out
fold -- the identical actor-disjoint-not-clip-disjoint principle applied
to real audio: segments from the same recording session are highly
correlated (same speaker, same room, same recording chain), so splitting
by segment would leak information the same way clip-splitting RAVDESS
would. The train fold's segments are added to training negatives (real-
world acoustic diversity BEATs would never see from RAVDESS alone); the
held-out fold is the honest real-negative generalization number. The
FULL 163-segment combined score is also reported separately, explicitly
labeled as including train-seen clips, purely for backward comparability
to the already-documented 0.018 max-score ceiling -- not presented as a
clean generalization measure.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

# Same documented bug as evaluate_full_system.py: real media filenames
# under real_recordings/ contain emoji/non-Latin characters the Windows
# console's default cp1252 encoding can't print. Fix at the source.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from beats_distress_detector import BEATsDistressDetector, SAMPLE_RATE
from calibrate_scream_detector import _load_pcm_16k_mono, collect_ravdess, collect_real_scream
from vad import VAD

REPO_ID = "birgermoell/ravdess"
TRAIN_ACTORS = list(range(1, 19))   # Actor_01..18
HELDOUT_ACTORS = list(range(19, 25))  # Actor_19..24, never seen in training
EMOTION_CODES = {"fear": "06", "angry": "05", "calm": "02", "happy": "03"}
INTENSITY = {"fear": "02", "angry": "02", "calm": "01", "happy": "01"}  # strong distress, normal control
STATEMENTS = ["01", "02"]
REPETITIONS = ["01", "02"]

REAL_NEGATIVES_DIR = Path(__file__).parent / "real_recordings" / "negatives"


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def download_ravdess(emotion: str, actor_nums: list[int]) -> list[bytes]:
    code, intensity = EMOTION_CODES[emotion], INTENSITY[emotion]
    clips = []
    for actor_num in actor_nums:
        actor = f"Actor_{actor_num:02d}"
        for statement in STATEMENTS:
            for repetition in REPETITIONS:
                fname = f"03-01-{code}-{intensity}-{statement}-{repetition}-{actor_num:02d}.wav"
                try:
                    local_path = hf_hub_download(REPO_ID, f"{actor}/{fname}", repo_type="dataset")
                except Exception:
                    continue
                clips.append(_load_pcm_16k_mono(local_path))
    return clips


def load_real_negatives_by_file() -> dict[str, list[bytes]]:
    """Real negative segments, keyed by source file -- so the train/
    held-out split can be made by FILE, not by segment (see module
    docstring for why)."""
    vad = VAD()
    by_file: dict[str, list[bytes]] = {}
    for path in sorted(REAL_NEGATIVES_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".mp3", ".wav", ".m4a"):
            continue
        try:
            audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()
        segs = [seg.pcm for seg in vad.segment_speech(pcm)]
        if segs:
            by_file[path.name] = segs
    return by_file


def extract_pooled_embedding(model: torch.nn.Module, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Frozen-backbone forward pass, stopping BEFORE the original
    AudioSet predictor -- returns the mean-pooled per-frame encoder
    hidden state (768-dim for this checkpoint). See module docstring's
    ARCHITECTURE CHANGE section for why pooling here is equivalent to
    the original architecture's pool-after-linear order."""
    audio = _pcm_to_float(pcm)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE // 10:
        return None
    source = torch.from_numpy(audio).unsqueeze(0)
    padding_mask = torch.zeros_like(source, dtype=torch.bool)
    with torch.no_grad():
        fbank = model.preprocess(source)
        padding_mask = model.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        features = model.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = model.layer_norm(features)
        if padding_mask is not None:
            padding_mask = model.forward_padding_mask(features, padding_mask)
        if model.post_extract_proj is not None:
            features = model.post_extract_proj(features)
        x = model.dropout_input(features)
        x, _layer_results = model.encoder(x, padding_mask=padding_mask)
        pooled = x.mean(dim=1)
    return pooled.squeeze(0).numpy()


def embed_all(model, clips: list[bytes]) -> np.ndarray:
    embeddings = [extract_pooled_embedding(model, pcm) for pcm in clips]
    return np.array([e for e in embeddings if e is not None])


def stats(name: str, scores) -> str:
    arr = np.array(scores)
    if len(arr) == 0:
        return f"  {name}: n=0"
    return f"  {name:40s} n={len(arr):3d} min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}"


def main() -> None:
    print("Loading BEATs (real backbone + original AudioSet head)...")
    detector = BEATsDistressDetector()
    beats_model = detector._model

    print(f"\nDownloading RAVDESS -- TRAIN actors {TRAIN_ACTORS[0]}-{TRAIN_ACTORS[-1]}, "
          f"HELD-OUT actors {HELDOUT_ACTORS[0]}-{HELDOUT_ACTORS[-1]} (24 actors total, verified all downloadable)...")
    train_fear = download_ravdess("fear", TRAIN_ACTORS)
    train_angry = download_ravdess("angry", TRAIN_ACTORS)
    train_calm = download_ravdess("calm", TRAIN_ACTORS)
    train_happy = download_ravdess("happy", TRAIN_ACTORS)
    heldout_fear = download_ravdess("fear", HELDOUT_ACTORS)
    heldout_angry = download_ravdess("angry", HELDOUT_ACTORS)
    heldout_calm = download_ravdess("calm", HELDOUT_ACTORS)
    heldout_happy = download_ravdess("happy", HELDOUT_ACTORS)
    print(f"  train: fear={len(train_fear)} angry={len(train_angry)} calm={len(train_calm)} happy={len(train_happy)}")
    print(f"  held-out: fear={len(heldout_fear)} angry={len(heldout_angry)} calm={len(heldout_calm)} happy={len(heldout_happy)}")

    print("\nLoading real_recordings/negatives, split by FILE (train fold vs held-out fold)...")
    by_file = load_real_negatives_by_file()
    files = sorted(by_file.keys())
    n_heldout_files = max(1, round(len(files) * 0.2))
    heldout_files = files[-n_heldout_files:]  # fixed, non-cherry-picked: last N by sorted filename
    train_files = files[:-n_heldout_files]
    train_real_neg = [seg for f in train_files for seg in by_file[f]]
    heldout_real_neg = [seg for f in heldout_files for seg in by_file[f]]
    all_real_neg = [seg for f in files for seg in by_file[f]]
    print(f"  {len(files)} files, {sum(len(v) for v in by_file.values())} total segments "
          f"(compare to the 163 segments used in this session's SER/BEATs FPR checks)")
    print(f"  train fold: {len(train_files)} files / {len(train_real_neg)} segments")
    print(f"  held-out fold: {len(heldout_files)} files / {len(heldout_real_neg)} segments -- {heldout_files}")

    print("\nExtracting frozen-backbone embeddings for all clips (no gradients, backbone untouched)...")
    X_train_pos = embed_all(beats_model, train_fear + train_angry)
    X_train_neg_ravdess = embed_all(beats_model, train_calm + train_happy)
    X_train_neg_real = embed_all(beats_model, train_real_neg)
    X_heldout_fear = embed_all(beats_model, heldout_fear)
    X_heldout_angry = embed_all(beats_model, heldout_angry)
    X_heldout_calm = embed_all(beats_model, heldout_calm)
    X_heldout_happy = embed_all(beats_model, heldout_happy)
    X_heldout_real_neg = embed_all(beats_model, heldout_real_neg)
    X_all_real_neg = embed_all(beats_model, all_real_neg)
    real_scream_pcm = collect_real_scream()
    # collect_real_scream() returns the DSP-calibration (1.95-12.93s) window --
    # too long for a single BEATs call without dilution (documented gotcha).
    # Re-slice to the correctly-windowed 2s peak established earlier this
    # session (45.0-47.0s of the raw clip) for both old and new heads.
    audio, sr = librosa.load(
        "real_recordings/positives/Nivetha Thomas Gets Abducted Vakeel Saab Malayalam Pawan Kalyan #YTShorts.mp3",
        sr=SAMPLE_RATE, mono=True,
    )
    start, end = int(45.0 * sr), int(47.0 * sr)
    real_scream_pcm = np.clip(audio[start:end] * 32767, -32768, 32767).astype(np.int16).tobytes()
    X_real_scream = embed_all(beats_model, [real_scream_pcm])

    X_train = np.vstack([X_train_pos, X_train_neg_ravdess, X_train_neg_real])
    y_train = np.array([1] * len(X_train_pos) + [0] * (len(X_train_neg_ravdess) + len(X_train_neg_real)))
    print(f"\nTraining set: {len(X_train)} clips ({len(X_train_pos)} distress / "
          f"{len(X_train_neg_ravdess)} RAVDESS-calm/happy / {len(X_train_neg_real)} real-negative)")

    print("Fitting logistic regression head (the new, small trainable layer)...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    head = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_train), y_train)

    def new_head_score(X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        return head.predict_proba(scaler.transform(X))[:, 1]

    def old_head_score(clips: list[bytes]) -> np.ndarray:
        return np.array([detector.score(pcm) for pcm in clips])

    print(f"\n{'=' * 96}\nBEFORE / AFTER -- identical clips, both heads (held-out actors 19-24, never used in training)\n{'=' * 96}")
    groups_heldout = {
        "HELD-OUT fear (n)": (heldout_fear, X_heldout_fear),
        "HELD-OUT angry (n)": (heldout_angry, X_heldout_angry),
        "HELD-OUT calm (n)": (heldout_calm, X_heldout_calm),
        "HELD-OUT happy (n)": (heldout_happy, X_heldout_happy),
        "HELD-OUT real negatives (n)": (heldout_real_neg, X_heldout_real_neg),
    }
    print(f"{'Group':<42}{'OLD head (AudioSet, 0-1)':>28}{'NEW head (fine-tuned, 0-1)':>28}")
    for label, (clips, X) in groups_heldout.items():
        old = old_head_score(clips)
        new = new_head_score(X)
        old_str = f"mean={old.mean():.3f} max={old.max():.3f}" if len(old) else "n=0"
        new_str = f"mean={new.mean():.3f} max={new.max():.3f}" if len(new) else "n=0"
        print(f"{label + f' n={len(clips)}':<42}{old_str:>28}{new_str:>28}")

    old_scream = old_head_score([real_scream_pcm])[0]
    new_scream = new_head_score(X_real_scream)[0]
    print(f"\n{'Real scream (correctly-windowed 45-47s)':<42}{old_scream:>28.4f}{new_scream:>28.4f}")

    old_all_real_neg = old_head_score(all_real_neg)
    new_all_real_neg = new_head_score(X_all_real_neg)
    print(f"\n{'ALL 163 real negatives (train+heldout combined -- NOT a clean generalization number, see below)':<0}")
    print(stats("  OLD head", old_all_real_neg))
    print(stats("  NEW head", new_all_real_neg))

    print(f"\n{'=' * 96}\nOVERFITTING CHECK -- NEW head, train actors (1-18, seen) vs held-out actors (19-24, unseen)\n{'=' * 96}")
    X_train_pos_scores = new_head_score(X_train_pos)
    X_heldout_pos_scores = new_head_score(np.vstack([X_heldout_fear, X_heldout_angry]))
    X_train_neg_scores = new_head_score(np.vstack([X_train_neg_ravdess]))
    X_heldout_neg_scores = new_head_score(np.vstack([X_heldout_calm, X_heldout_happy]))
    print(f"  TRAIN distress (fear+angry, seen actors):     n={len(X_train_pos_scores)} mean={X_train_pos_scores.mean():.4f} min={X_train_pos_scores.min():.4f}")
    print(f"  HELD-OUT distress (fear+angry, unseen actors): n={len(X_heldout_pos_scores)} mean={X_heldout_pos_scores.mean():.4f} min={X_heldout_pos_scores.min():.4f}")
    print(f"  TRAIN calm/happy (seen actors):                n={len(X_train_neg_scores)} mean={X_train_neg_scores.mean():.4f} max={X_train_neg_scores.max():.4f}")
    print(f"  HELD-OUT calm/happy (unseen actors):           n={len(X_heldout_neg_scores)} mean={X_heldout_neg_scores.mean():.4f} max={X_heldout_neg_scores.max():.4f}")
    gap = X_train_pos_scores.mean() - X_heldout_pos_scores.mean()
    print(f"\n  Train-vs-held-out mean-score gap on distress clips: {gap:+.4f} "
          f"({'meaningful overfitting signal' if gap > 0.15 else 'small, within normal train/val variance'})")

    print(f"\n{'=' * 96}\nTHRESHOLD SWEEP -- NEW head, held-out actors' fear+angry vs held-out real negatives (0% FPR discipline)\n{'=' * 96}")
    heldout_pos_scores = new_head_score(np.vstack([X_heldout_fear, X_heldout_angry]))
    heldout_neg_scores = new_head_score(X_heldout_real_neg) if len(X_heldout_real_neg) else np.array([])
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        recall = np.mean(heldout_pos_scores >= t) if len(heldout_pos_scores) else 0.0
        fpr = np.mean(heldout_neg_scores >= t) if len(heldout_neg_scores) else 0.0
        print(f"  threshold={t:.2f}  recall={recall:.1%}  FPR(held-out real negatives)={fpr:.1%}")

    print(f"\n{'=' * 96}\nMANDATORY SCOPE STATEMENT\n{'=' * 96}")
    print("  This fine-tune improves recognition of RAVDESS-style ACTED distress specifically.")
    print("  It does NOT constitute training on genuine real-world assault/threat audio, which")
    print("  remains unavailable. It narrows the acted-distress-detection ceiling -- it does NOT")
    print("  close the 'no real distress data' gap documented elsewhere in this project.")

    import pickle

    with open("beats_finetuned_head.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "head": head}, f)
    print("\nSaved trained head + scaler to beats_finetuned_head.pkl (NOT loaded by beats_distress_detector.py -- standalone artifact only).")


if __name__ == "__main__":
    main()
