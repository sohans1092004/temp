"""Keyword-spotting distances only -- no speech-to-text, no Parakeet (GPU, ~20-30 min).
    python kws_distances.py --root /path/to/real_world
Dumps, per VAD segment, KWS's best (phrase, DTW distance) with the match threshold opened
up, for (a) every prepped clip x 6 conditions and (b) the ambient + hard YouTube files,
chunked exactly like evaluate_real_world.py so times line up with real_world_alerts.csv.
-> kws_distances.json. Used to pick a KWS-alone threshold that keeps danger detections."""
import json
import sys
from pathlib import Path

import numpy as np

import evaluate_prepped_data as ep
from evaluate_real_world import SR, arg, decode, files

CONDS = ["normal", "quiet", "loud", "crowd", "pocket", "phone"]


def main() -> None:
    from evaluate_full_system import default_device
    from kws import Wav2Vec2DTWSpotter
    from vad import VAD

    root = Path(arg("--root", "real_world"))
    kws = Wav2Vec2DTWSpotter(device=default_device(), match_threshold=10.0)  # report every best match
    kws.load_references(Path(__file__).parent / "kws_references.npz")
    vad = VAD()

    def segs(pcm, offset=0.0):
        out = []
        for s in vad.segment_speech(pcm):
            m = kws.spot(s.pcm)
            if m is not None:
                out.append({"t": round(offset + s.start_ms / 1000, 1), "udk": m.phrase_id, "d": round(m.distance, 4)})
        return out

    result = {"prepped": [], "real": []}
    for c in CONDS:
        cond = ep.condition(c)
        for p in ep.test_clips():
            result["prepped"].append({"clip": p.stem, "cond": c, "want": sorted(ep.expected_udks(p.name)),
                                      "segs": segs(cond(ep.load_pcm(p)))})
        print(f"prepped {c} done", flush=True)
    for path in files(root):
        if path.parent.name not in ("ambient", "hard"):
            continue
        audio = decode(path)[: int(45 * 60 * SR)]
        for k in range(int(np.ceil(len(audio) / (5 * 60 * SR)))):
            a = audio[k * 5 * 60 * SR: (k + 1) * 5 * 60 * SR]
            pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
            result["real"].append({"category": path.parent.name, "file": path.stem, "segs": segs(pcm, k * 5 * 60)})
        print(f"{path.parent.name}/{path.name} done", flush=True)
    Path("kws_distances.json").write_text(json.dumps(result))
    print("-> kws_distances.json", flush=True)


if __name__ == "__main__":
    main()
