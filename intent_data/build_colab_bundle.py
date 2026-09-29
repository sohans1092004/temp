"""Builds <Model>/udk_intent_colab.zip for intent_data.finetune on Colab: the dataset, the
intent scripts, the matching engine the regression replay needs, and the prepped Colab
logs (copied from ~/Downloads into intent_data/logs/). No audio, no model weights.
    python intent_data/build_colab_bundle.py"""
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = HERE.parent
ROOT = "udk_intent"
ENGINE = ["udk_engine.py", "udks.py", "semantic.py", "kws.py", "beats_distress_detector.py",
          "intent_gate.py", "stt.py"]  # intent_gate: premise() for finetune; stt: harvest_misheard
LOGS = [Path.home() / "Downloads" / f"prepped_gpu_{c}.log" for c in ("normal", "quiet", "loud", "crowd", "pocket", "phone")]
REQUIREMENTS = """\
# Colab already has torch (CUDA), transformers, numpy, scikit-learn, huggingface_hub.
sentence-transformers   # MiniLM semantic matcher (regression replay) + the baseline classifier
rapidfuzz               # fuzzy matching (regression replay)
sentencepiece           # DeBERTa-v3 / mDeBERTa tokenizers
protobuf
accelerate              # device_map for llm_eval.py; sentence-transformers fit()
datasets                # sentence-transformers fit() (finetune_minilm.py)
"""
HARVEST_REQUIREMENTS = """# harvest_misheard.py only: the two live STTs (then swap in onnxruntime-gpu, as in the GPU package README)
faster-whisper
onnx-asr[hub]==0.12.0
soundfile
librosa
"""


def main() -> None:
    out = PROJECT / "udk_intent_colab.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.suffix in {".py", ".csv", ".txt"}:
                z.write(p, f"{ROOT}/intent_data/{p.name}")
        for sub, pattern in (("llm", "*.jsonl"), ("misheard", "*.csv")):  # build_v2 inputs, if already made
            for p in sorted((HERE / sub).glob(pattern)):
                z.write(p, f"{ROOT}/intent_data/{sub}/{p.name}")
        for p in sorted((HERE / "real_world").glob("*.csv")):  # real-world alert lists (realworld_gate.py)
            z.write(p, f"{ROOT}/intent_data/real_world/{p.name}")
        for f in ENGINE:
            z.write(PROJECT / f, f"{ROOT}/{f}")
        missing = [p.name for p in LOGS if not p.exists()]
        for p in LOGS:
            if p.exists():
                z.write(p, f"{ROOT}/intent_data/logs/{p.name}")
        z.writestr(f"{ROOT}/requirements_intent.txt", REQUIREMENTS)
        z.writestr(f"{ROOT}/requirements_harvest.txt", HARVEST_REQUIREMENTS)
    print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB" + (f"  (missing logs: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
