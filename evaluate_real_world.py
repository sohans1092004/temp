"""False alarms per hour on real, long, non-distress recordings (real_world/), for the
live configuration: Whisper + Parakeet UDK detection (KWS, matching, per-sentence) and
the scream-alone trigger (scream_trigger.py). TEST ONLY -- these recordings never train.

    python evaluate_real_world.py run    [--root real_world] [--max-min 45] [--chunk-min 5]
    python evaluate_real_world.py report [--root real_world]
    add --sealed to both on a SEALED test set: progress and report show totals only (per
    category + overall), no file names, transcripts or alert list -- safe to paste.

Layout: <root>/{ambient,speech,hard}/<audio files> + <root>/videos.csv (URL, Title, ...).
Every alert on these recordings is a false alarm by construction (nothing dangerous in
them). `run` works in chunks and saves after each one: re-run the same command after a
disconnect and it resumes. `report` writes real_world_report.md + real_world_alerts.csv
(every alert with a YouTube link at its timestamp, for review).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).parent
SR = 16000
AUDIO = {".mp3", ".mpeg", ".m4a", ".webm", ".opus", ".wav", ".mp4", ".ogg", ".flac", ".aac"}
REFRACTORY_S = 10.0  # alerts closer than this count as one alert episode
SEALED = "--sealed" in sys.argv


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*") if p.suffix.lower() in AUDIO)


def urls(root: Path) -> dict[str, str]:
    """file -> YouTube URL, by fuzzy-matching file names to the CSV's Title column."""
    from rapidfuzz import fuzz, process

    csv_path = root / "videos.csv"
    if not csv_path.exists():
        return {}
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")) if r.get("URL")]
    titles = {r["Title"].strip().lower(): r["URL"].strip() for r in rows if r.get("Title")}
    # an explicit `filename` column (e.g. "hard/gameplay.mp3") wins over fuzzy title matching
    by_name = {Path(r["filename"].strip()).as_posix(): r["URL"].strip() for r in rows if (r.get("filename") or "").strip()}
    out = {}
    for p in files(root):
        rel = p.relative_to(root).as_posix()
        if rel in by_name:
            out[str(p.relative_to(root))] = by_name[rel]
            continue
        hit = process.extractOne(p.stem.lower(), list(titles), scorer=fuzz.WRatio) if titles else None
        if hit and hit[1] >= 75:
            out[str(p.relative_to(root))] = titles[hit[0]]
    return out


def load_models():
    from beats_distress_detector import BEATsDistressDetector
    from evaluate_full_system import default_device
    from kws import Wav2Vec2DTWSpotter
    from semantic import SentenceTransformerSemanticMatcher
    from stt import FasterWhisperSTT, ParakeetSTT
    from vad import VAD

    import intent_gate

    dev = default_device()
    kws = None
    if os.environ.get("UDK_ENABLE_KWS", "1") != "0":  # UDK_ENABLE_KWS=0: the deployed config (KWS off)
        kws = Wav2Vec2DTWSpotter(device=dev)
        kws.load_references(HERE / "kws_references.npz")
    m = {"vad": VAD(), "kws": kws, "semantic": SentenceTransformerSemanticMatcher(device=dev),
         "beats": BEATsDistressDetector(device=dev),
         "stts": {"whisper": FasterWhisperSTT(model_size="small.en", device=dev, compute_type="float16" if dev == "cuda" else "int8"),
                  "parakeet": ParakeetSTT(device=dev)},
         "gate": intent_gate.from_env(dev)}
    print(f"device: {dev}; Parakeet on {m['stts']['parakeet'].providers[0]}", flush=True)
    return m


def decode(path: Path) -> np.ndarray:
    from faster_whisper import decode_audio

    return decode_audio(str(path), sampling_rate=SR)


def run(root: Path, max_min: float, chunk_min: float) -> None:
    import scream_trigger
    from evaluate_full_system import run_file_either

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    todo = files(root)
    print(f"{len(todo)} files; cap {max_min} min each; {chunk_min}-min chunks", flush=True)
    models = None
    t_start, done_audio = time.time(), 0.0
    for path in todo:
        cache_f = out_dir / f"{path.parent.name}__{path.stem}.json"
        cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {"chunks": {}}
        audio = None
        n_chunks = None
        for k in range(10_000):
            if str(k) in cache["chunks"]:
                continue
            if audio is None:
                audio = decode(path)[: int(max_min * 60 * SR)]
                n_chunks = int(np.ceil(len(audio) / (chunk_min * 60 * SR)))
            if k >= n_chunks:
                break
            if models is None:
                models = load_models()
            a = audio[int(k * chunk_min * 60 * SR): int((k + 1) * chunk_min * 60 * SR)]
            pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
            offset = k * chunk_min * 60
            runs = run_file_either(pcm, models["vad"], models["stts"], models["kws"], models["semantic"], models["beats"],
                                   gate=models["gate"])
            udk = [{"t": round(offset + r["start_ms"] / 1000, 1), "stt": name, "decision": r["decision"], "udk": r["udk_id"],
                    "layer": r["layer"], "conf": round(r["confidence"], 3), "text": r["transcript"], "kws": r.get("kws")}
                   for name, (_, rows, _) in runs.items() for r in rows if r["decision"] != "NO_ACTION"]
            scream = []
            for seg in models["vad"].segment_speech(pcm):
                fired, peak, cls = scream_trigger.check(models["beats"], seg.pcm)
                if fired:
                    scream.append({"t": round(offset + seg.start_ms / 1000, 1), "score": round(peak, 3), "class": cls})
            cache["chunks"][str(k)] = {"seconds": len(a) / SR, "udk": udk, "scream": scream}
            cache_f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            done_audio += len(a) / SR
            el = time.time() - t_start
            if SEALED:
                print(f"  {done_audio / 3600:.2f} h audio in {el / 60:.1f} min", flush=True)
            else:
                print(f"  {path.parent.name}/{path.name} chunk {k + 1}/{n_chunks}: {len(udk)} UDK rows, {len(scream)} screams "
                      f"| {done_audio / 3600:.2f} h audio in {el / 60:.1f} min (RTF {el / done_audio:.2f})", flush=True)
    print("done", flush=True)


def episodes(times: list[float]) -> int:
    n, last = 0, -1e9
    for t in sorted(times):
        if t - last >= REFRACTORY_S:
            n += 1
        last = t
    return n


def report(root: Path) -> None:
    link = urls(root)
    out, alerts = [], []
    per_cat = {}
    rows_md = []
    for cache_f in sorted((root / "results").glob("*.json")):
        cat, stem = cache_f.stem.split("__", 1)
        c = json.loads(cache_f.read_text(encoding="utf-8"))["chunks"]
        hours = sum(v["seconds"] for v in c.values()) / 3600
        udk = [u for v in c.values() for u in v["udk"]]
        scream = [s for v in c.values() for s in v["scream"]]
        t_w = [u["t"] for u in udk if u["stt"] == "whisper"]
        t_p = [u["t"] for u in udk if u["stt"] == "parakeet"]
        t_s = [s["t"] for s in scream]
        rel = next((k for k in link if Path(k).parent.name == cat and Path(k).stem == stem), None)
        url = link.get(rel, "")
        e = {"w": episodes(t_w), "p": episodes(t_p), "wp": episodes(t_w + t_p), "s": episodes(t_s), "all": episodes(t_w + t_p + t_s)}
        agg = per_cat.setdefault(cat, {"h": 0.0, "files": 0, "hit": 0, **{k: 0 for k in e}})
        agg["h"] += hours
        agg["files"] += 1
        agg["hit"] += e["all"] > 0  # recall sets (911 calls): a file counts once it alerted at all
        for k, v in e.items():
            agg[k] += v
        rows_md.append(f"| {cat} | {stem} | {hours * 60:.0f} | " + " | ".join(f"{e[k] / hours:.1f}" if hours else "-" for k in ("w", "p", "wp", "s", "all"))
                       + f" | {'yes' if url else 'no'} |")
        for u in udk:
            alerts.append({"category": cat, "file": stem, "time": time.strftime("%H:%M:%S", time.gmtime(u["t"])),
                           "rule": f"UDK {u['udk']} ({u['layer']} {u['conf']}, {u['stt']})"
                           + (f" [kws {u['kws'][0]} d={u['kws'][1]}]" if u.get("kws") else ""), "tier": u["decision"],
                           "heard": u["text"], "link": f"{url}&t={int(u['t'])}s" if url else ""})
        for s in scream:
            alerts.append({"category": cat, "file": stem, "time": time.strftime("%H:%M:%S", time.gmtime(s["t"])),
                           "rule": f"scream-alone ({s['class']} {s['score']})", "tier": "TRIGGER_VERIFY", "heard": "",
                           "link": f"{url}&t={int(s['t'])}s" if url else ""})
    out += ["# False alarms on real non-distress recordings", "",
            "Every alert here is a false alarm (the recordings contain no real danger). Rates are alert "
            f"**episodes per hour** (alerts within {REFRACTORY_S:.0f} s merged). TEST ONLY data.", "",
            "## Per category", "",
            "| Category | Hours | Whisper only | Parakeet only | Whisper + Parakeet | Scream-alone | **Everything live** | Files alerting |",
            "|---|---|---|---|---|---|---|---|"]
    for cat, a in per_cat.items():
        out.append(f"| {cat} | {a['h']:.2f} | " + " | ".join(f"{a[k] / a['h']:.1f}" if a["h"] else "-" for k in ("w", "p", "wp", "s"))
                   + f" | **{a['all'] / a['h']:.1f}** | {a['hit']}/{a['files']} |" if a["h"] else f"| {cat} | 0 | - | - | - | - | - | - |")
    tot = {k: sum(a[k] for a in per_cat.values()) for k in ("h", "files", "hit", "w", "p", "wp", "s", "all")}
    if tot["h"]:
        out.append("| **ALL** | " + f"{tot['h']:.2f} | " + " | ".join(f"{tot[k] / tot['h']:.1f}" for k in ("w", "p", "wp", "s"))
                   + f" | **{tot['all'] / tot['h']:.1f}** | {tot['hit']}/{tot['files']} |")
    if SEALED:  # totals only: nothing that names a file, a time or what was heard
        print("\n".join(out[:2] + out[4:]))
        return
    out += ["", "## Per file (episodes per hour)", "",
            "| Category | File | Minutes | Whisper | Parakeet | W + P | Scream | All | URL matched |", "|---|---|---|---|---|---|---|---|---|"] + rows_md
    out += ["", f"Every alert, with a YouTube link at its timestamp: `real_world_alerts.csv` ({len(alerts)} rows)."]
    (root / "real_world_report.md").write_text("\n".join(out), encoding="utf-8")
    with open(root / "real_world_alerts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "file", "time", "rule", "tier", "heard", "link"])
        w.writeheader()
        w.writerows(sorted(alerts, key=lambda a: (a["category"], a["file"], a["time"])))
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    root = Path(arg("--root", str(HERE / "real_world")))
    if cmd == "run":
        run(root, arg("--max-min", 45.0), arg("--chunk-min", 5.0))
    elif cmd == "report":
        report(root)
    else:
        sys.exit(__doc__)
