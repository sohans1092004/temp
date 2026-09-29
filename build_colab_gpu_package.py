"""Builds udk_colab_gpu.zip: everything evaluate_prepped_data.py and the
real-recordings part of evaluate_full_system.py need, to run Whisper
small.en on a Colab GPU. Models (Whisper, wav2vec2, MiniLM, BEATs) download
from HuggingFace inside Colab, so no model weights are shipped.
Re-run after changing any of the listed files."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path


def kaggle_safe(name: str) -> str:
    """Kaggle rejects dataset file names with '#', quotes, emoji... (YouTube-titled recordings);
    renamed inside the zip only -- code finds recordings by folder, never by name."""
    return re.sub(r"[^\w.\-/ ()]", "_", name)

HERE = Path(__file__).parent
ROOT = "udk_colab_gpu"
CODE = [
    "audio_augment.py", "beats_distress_detector.py", "evaluate_full_system.py",
    "evaluate_pipeline_corpus.py", "evaluate_prepped_data.py", "kws.py", "semantic.py",
    "separation.py", "stt.py", "stt_confidence_gate.py", "udk_engine.py", "udks.py",
    "vad.py", "wesep_extraction.py", "kws_references.npz",
    # live-path check (colab_api_check.py): api.py and everything it imports
    "colab_api_check.py", "api.py", "audit.py", "crypto.py", "db.py", "db_postgres.py",
    "dual_asr_guard.py", "enrollment.py", "event_delivery.py", "kafka_delivery.py",
    "language_id.py", "metrics.py", "storage.py", "storage_s3.py", "scream_trigger.py", "intent_gate.py", "evaluate_real_world.py", "kws_distances.py",
]
DIRS = ["beats_vendor", "prepped_data", "real_recordings"]
SKIP_SUFFIXES = {".pyc", ".zip"}  # real_recordings/panic_phrases_synthetic_audio.zip isn't used by any eval
EXTRA = {"requirements_colab.txt": "colab_requirements.txt", "README.md": "colab_README.md"}


def main() -> None:
    out = HERE / "udk_colab_gpu.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(HERE / f, f"{ROOT}/{f}")
            n += 1
        for d in DIRS:
            for p in sorted((HERE / d).rglob("*")):
                # user_audio/ = the enrolled user's voice (biometric), only used by
                # WeSep, which this package doesn't include -- don't ship it.
                if (p.is_file() and p.suffix.lower() not in SKIP_SUFFIXES
                        and "__pycache__" not in p.parts and "user_audio" not in p.parts):
                    z.write(p, kaggle_safe(f"{ROOT}/{p.relative_to(HERE).as_posix()}"))
                    n += 1
        for arcname, src in EXTRA.items():
            z.write(HERE / src, f"{ROOT}/{arcname}")
            n += 1
    print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
