"""Held-out TEST ONLY: runs every clip in prepped_data/ through the current
Model detection path (evaluate_full_system.run_file: VAD -> STT tiny.en +
KWS -> BEATs -> stt_confidence_gate -> UDKEngine with fuzzy + semantic
MiniLM@0.65). Nothing here trains, calibrates or tunes on this data.

STT is Whisper + Parakeet by default (evaluate_full_system.run_file_either):
both run on the same VAD segments and a clip counts as detected if EITHER
alerts (the stronger tier wins). PREPPED_PARAKEET=0 = Whisper only, the
pre-2026-09-24 configuration, for comparison.

Ground truth comes from each filename: every "UDK_NN" it names that is not
under a "NOT a UDK" label is expected to fire; a file naming no expected
UDK is a control and should fire nothing. WeSep is not used (no voice
sample of these speakers exists to enroll).

Clips: every audio file directly in prepped_data/, .mp4 included (the real
traffic recordings, labelled by filename like the rest). Files named only
with a number (1.mp4 ... 6.mp4) are skipped: unlabelled, byte-identical
copies of the named ones, and they'd count as controls.

PREPPED_CONDITION=normal|quiet|loud|crowd|pocket|phone re-runs the same clips
with the audio altered (quiet -12 dB, loud +12 dB with clipping, crowd noise
at 10 dB SNR, pocket muffling, phone-band 300-3400 Hz), the robustness set
used for the 2026-09-24 separator/Parakeet evaluation. Default: normal.
"""
from __future__ import annotations

import os
import time
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate_full_system import load_real_pcm, run_file_either
from udks import GENERAL_UDKS

DATA = Path(__file__).parent / "prepped_data"
USER_ENROLLMENT = DATA / "user_audio" / "keerthana_audio.ogg"
AUDIO_SUFFIXES = {".mp3", ".mpeg", ".ogg", ".wav", ".m4a", ".flac"}
# Per the dataset owner: every clip contains the enrolled user EXCEPT these two.
NO_USER = {
    "Calm, ordinary disagreement, no UDK 2 females one by one",
    "Casual chat, overlapping, no UDK 2 females one by one",
}
PHRASE = {u.udk_id: u.phrase for u in GENERAL_UDKS}


def load_pcm(path: Path) -> bytes:
    """3 of the 17 files are AAC (ADTS) saved with a .mp3 name, which
    librosa/libsndfile can't open -- decode those with faster-whisper's own
    PyAV decoder (already installed, no new dependency)."""
    try:
        return load_real_pcm(path)
    except Exception:
        import numpy as np
        from faster_whisper import decode_audio

        audio = decode_audio(str(path), sampling_rate=16000)
        return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


def expected_udks(name: str) -> set[str]:
    return set() if "NOT" in name else set(re.findall(r"UDK_\d\d", name))


def test_clips() -> list[Path]:
    """Every test clip directly in prepped_data/ (user_audio/ is a subfolder:
    the enrollment voice, not a test clip)."""
    clips = []
    for p in sorted(DATA.glob("*")):
        if p.suffix.lower() not in AUDIO_SUFFIXES | {".mp4"}:
            continue
        if p.stem.isdigit():  # 1.mp4 ... 6.mp4: unlabelled duplicates of the named copies
            print(f"(skipping {p.name}: unlabelled duplicate, would count as a control)")
            continue
        clips.append(p)
    return clips


def _phone(pcm: bytes) -> bytes:
    from scipy.signal import butter, filtfilt, resample_poly

    from audio_augment import _to_float, _to_pcm

    x = resample_poly(_to_float(pcm), 1, 2)  # 8 kHz
    b, a = butter(4, [300 / 4000, 3400 / 4000], btype="band")
    return _to_pcm(resample_poly(filtfilt(b, a, x), 2, 1))


def condition(name: str):
    """pcm -> pcm for PREPPED_CONDITION (see module docstring)."""
    from audio_augment import _to_float, _to_pcm, add_noise, muffle

    return {
        "normal": lambda p: p,
        "quiet": lambda p: _to_pcm(_to_float(p) * 10 ** (-12 / 20)),
        "loud": lambda p: _to_pcm(_to_float(p) * 10 ** (12 / 20)),  # _to_pcm clips
        "crowd": lambda p: add_noise(p, 16000, snr_db=10, kind="crowd"),
        "pocket": lambda p: muffle(p, 16000),
        "phone": _phone,
    }[name]


def main() -> None:
    from beats_distress_detector import BEATsDistressDetector
    from kws import Wav2Vec2DTWSpotter
    from semantic import SentenceTransformerSemanticMatcher
    from stt import FasterWhisperSTT, ParakeetSTT
    from vad import VAD

    # PREPPED_STT_MODEL: default small.en (preferred config).
    # PREPPED_DEVICE=cuda: EVERY model on GPU -- Whisper, KWS wav2vec2, BEATs,
    # semantic MiniLM (and WeSep/ECAPA if enabled). PREPPED_STT_DEVICE is the
    # older Whisper-only switch, still honoured as a fallback.
    model = os.environ.get("PREPPED_STT_MODEL", "small.en")
    from evaluate_full_system import default_device

    device = os.environ.get("PREPPED_DEVICE", os.environ.get("PREPPED_STT_DEVICE", default_device()))
    # PREPPED_PARAKEET=0: Whisper only (the pre-2026-09-24 config), for comparison.
    use_parakeet = os.environ.get("PREPPED_PARAKEET", "1") != "0"
    cond_name = os.environ.get("PREPPED_CONDITION", "normal")
    alter = condition(cond_name)
    print(f"Device for all models: {device}   STT: faster-whisper {model}"
          + (" + parakeet (alert if either)" if use_parakeet else "") + f"   CONDITION: {cond_name}")
    stt = FasterWhisperSTT(model_size=model, device=device, compute_type="float16" if device == "cuda" else "int8")
    stts = {"whisper": stt}
    if use_parakeet:
        stts["parakeet"] = ParakeetSTT(device=device)
        print(f"Parakeet running on: {stts['parakeet'].providers[0]}")
    vad = VAD()
    kws = None
    if os.environ.get("PREPPED_KWS", "1") != "0":  # PREPPED_KWS=0: live config with UDK_ENABLE_KWS=0
        kws = Wav2Vec2DTWSpotter(device=device)
        kws.load_references(Path(__file__).parent / "kws_references.npz")
    print(f"keyword spotting: {'on' if kws else 'OFF'}")
    semantic = SentenceTransformerSemanticMatcher(device=device)
    beats = BEATsDistressDetector(device=device)
    import intent_gate

    gate = intent_gate.from_env(device)  # UDK_INTENT_GATE=<mDeBERTa dir>: veto-agree on text alerts

    timings = {}  # component -> total seconds, across all clips

    def timed(obj, method: str, component: str) -> None:
        """Wraps obj.method so its wall time adds up under `component`."""
        inner = getattr(obj, method)

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                timings[component] = timings.get(component, 0.0) + time.perf_counter() - t0
        setattr(obj, method, wrapper)

    timed(stt, "transcribe", "Whisper STT")
    if use_parakeet:
        timed(stts["parakeet"], "transcribe", "Parakeet STT")
    if kws is not None:
        timed(kws, "_frame_embeddings", "KWS wav2vec2")
        timed(kws, "spot", "KWS total (wav2vec2 + DTW)")
    timed(beats, "score", "BEATs")
    timed(semantic, "similarity", "Semantic MiniLM")

    # PREPPED_WESEP=1: WeSep target-speaker extraction (NO_ACTION fallback +
    # weak-VERIFY re-check), enrolled with the user's own voice. The two
    # NO_USER clips don't contain the user, so they run without it. The
    # self-check threshold is NOT re-tuned on this held-out data.
    wesep = None
    if os.environ.get("PREPPED_WESEP") == "1":
        from evaluate_wesep_overlap import LoggingSeparator
        from wesep_extraction import EXTRACTION_ACCEPT_THRESHOLD, WeSepTargetSeparator

        wesep = LoggingSeparator(WeSepTargetSeparator(load_pcm(USER_ENROLLMENT), device=device))
        timed(wesep.inner, "separate", "WeSep (extract + self-check)")
        print(f"WeSep ON: enrolled {USER_ENROLLMENT.name}, self-check threshold {EXTRACTION_ACCEPT_THRESHOLD}")

    summary, outcomes, clip_times = [], [], []
    for path in test_clips():
        want = expected_udks(path.name)
        separator = wesep if wesep is not None and path.stem not in NO_USER else None
        scores_before = len(wesep.scores) if wesep is not None else 0
        t0 = time.monotonic()
        runs = run_file_either(alter(load_pcm(path)), vad, stts, kws, semantic, beats, separator, gate=gate)
        latency = time.monotonic() - t0
        # Either path alerting counts: pooling both paths' events makes the clip's
        # tier the stronger of the two, and its UDKs the union.
        events = [e for path_events, _, _ in runs.values() for e in path_events]
        rows = [{**r, "stt": name} for name, (_, path_rows, _) in runs.items() for r in path_rows]
        wesep_scores = wesep.scores[scores_before:] if wesep is not None else []
        clip_times.append(latency)
        tier = "TRIGGER_ALL" if any(e.decision == "TRIGGER_ALL" for e in events) else (
            "TRIGGER_VERIFY" if events else "nothing")
        outcomes.append((path.stem, bool(want), tier, max((e.confidence for e in events), default=None)))
        got = {e.udk.udk_id for e in events if e.udk}
        print(f"\n{'=' * 100}\n{path.stem}\n  expected: "
              + (", ".join(f"{u} ({PHRASE[u]!r})" for u in sorted(want)) or "nothing (control)"))
        for r in rows:
            out = "NO_ACTION" if r["decision"] == "NO_ACTION" else f"{r['decision']} {r['udk_id']} ({r['layer']}, conf {r['confidence']:.2f})"
            by = f" [{r['stt']}]" if len(stts) > 1 else ""
            k = f"   [kws {r['kws'][0]} d={r['kws'][1]}]" if r.get("kws") else ""
            v = f"   [intent veto: {r['udk_id']} {r['layer']}, p={r['veto_p']}]" if "veto_p" in r else ""
            print(f"  heard{by}: {r['transcript']!r}\n     -> {out}{k}{v}")
        if not rows:
            print("  heard: (no speech transcribed)")
        if wesep is not None:
            print("  WeSep: " + ("not used (user not in this clip)" if separator is None else
                                 ("self-check scores " + ", ".join(f"{s:.3f}" for s in wesep_scores)) if wesep_scores
                                 else "not needed (every segment already matched exactly)"))
        if want:
            hit, missed, extra = want & got, want - got, got - want
            verdict = "ALL FOUND" if not missed else ("PARTIAL" if hit else "MISSED")
        else:
            hit, missed, extra = set(), set(), got
            verdict = "CORRECT (silent)" if not got else "FALSE ALARM"
        print(f"  RESULT: {verdict}  found={sorted(hit)} missed={sorted(missed)} other={sorted(extra)}  ({latency:.1f}s)")
        summary.append((path.stem, want, hit, missed, extra, verdict))

    print(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    for stem, want, hit, missed, extra, verdict in summary:
        print(f"  {verdict:<17} {stem[:75]}")
    udk_files = [s for s in summary if s[1]]
    controls = [s for s in summary if not s[1]]
    n_want = sum(len(s[1]) for s in udk_files)
    n_hit = sum(len(s[2]) for s in udk_files)
    print(f"\n  UDK instances detected: {n_hit}/{n_want} across {len(udk_files)} files "
          f"({sum(s[5] == 'ALL FOUND' for s in udk_files)} files fully found)")
    print(f"  Control files that false-alarmed: {sum(s[5] == 'FALSE ALARM' for s in controls)}/{len(controls)}")
    print(f"  Wrong extra UDKs fired inside UDK files: {sum(len(s[4]) for s in udk_files)}")

    print(f"\n{'=' * 100}\nALERT-LEVEL RESULT (TRIGGER_ALL or TRIGGER_VERIFY = success, nothing = no detection)\n{'=' * 100}")
    for stem, danger, tier, conf in outcomes:
        c = f"{conf:.2f} (+{conf - 0.60:.2f})" if conf is not None else "-"
        print(f"  {'DANGER ' if danger else 'control'}  {tier:<15} conf {c:<14} {stem[:70]}")
    tp = sum(d and t != "nothing" for _, d, t, _ in outcomes)
    fn = sum(d and t == "nothing" for _, d, t, _ in outcomes)
    fp = sum(not d and t != "nothing" for _, d, t, _ in outcomes)
    tn = sum(not d and t == "nothing" for _, d, t, _ in outcomes)
    n_all = sum(d and t == "TRIGGER_ALL" for _, d, t, _ in outcomes)
    print(f"\n  danger clips: {tp + fn}  (TRIGGER_ALL {n_all}, TRIGGER_VERIFY {tp - n_all}, nothing {fn})")
    print(f"  control clips: {fp + tn}  (alerted {fp}, silent {tn})")
    print(f"  Recall    {tp}/{tp + fn} = {tp / max(tp + fn, 1):.1%}")
    print(f"  FPR       {fp}/{fp + tn} = {fp / max(fp + tn, 1):.1%}")
    print(f"  Accuracy  {tp + tn}/{len(outcomes)} = {(tp + tn) / max(len(outcomes), 1):.1%}")
    print(f"  Precision {tp}/{tp + fp} = {tp / max(tp + fp, 1):.1%}")
    n = len(clip_times)
    print(f"\n  Timing per clip ({model}, all models on {device}), mean over {n} clips:")
    for comp, total in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"    {comp:<32} {total / n:6.2f} s")
    # "KWS wav2vec2" is nested inside "KWS total". The Whisper/KWS runs WeSep
    # triggers on the extracted voice are counted under Whisper/KWS, not WeSep.
    accounted = sum(v for k, v in timings.items() if k != "KWS wav2vec2")
    print(f"    {'rest (VAD, matching, overhead)':<32} {max(sum(clip_times) - accounted, 0) / n:6.2f} s")
    print(f"    {'WHOLE PIPELINE':<32} {sum(clip_times) / n:6.2f} s   (max {max(clip_times):.2f} s)")


if __name__ == "__main__":
    main()
