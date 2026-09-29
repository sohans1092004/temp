"""Full-system evaluation: the current shipped, recommended production
configuration (VAD -> STT+KWS in parallel -> BEATs scream corroboration
-> stt_confidence_gate veto -> UDKEngine.decide()) run against every
corpus this project has -- the synthetic TTS+augmentation corpus AND
every folder under real_recordings/ (mixed, positives, negatives) -- in
one pass, reporting correct/false-positive/false-negative counts and
latency for each corpus and for the system as a whole.

Deliberately excludes dual_asr_guard.py (UDK_ENABLE_DUAL_ASR) -- per its
own module docstring, a real 25% false-veto rate makes it explicitly not
recommended for anything resembling production use. separation.py's
NO_ACTION-only fallback is included but OFF by default (set
UDK_ENABLE_SEPARATION=1 to include it) since it only ever fires after an
already-missed segment and adds real per-retry latency (~4-8s) -- kept
optional so the default run reflects the fast-path latency profile.

Ground truth per corpus, stated explicitly since they differ:
  - Synthetic corpus (evaluate_pipeline_corpus.py's own builder): exact
    UDK id expected per positive clip; NO_ACTION expected for negatives.
  - real_recordings/mixed: manifest.json gives the exact phrase spoken
    per clip. Where that phrase is byte-identical to a real UDK phrase,
    exact-match correctness is required (matched_id == expected AND
    decision != NO_ACTION). Where the manifest phrase is a paraphrase not
    in the 20-UDK list (most of them), correctness is scored as
    detected-something (decision != NO_ACTION) -- the same "recall
    proxy" bar this project has used for real, non-exact-phrase audio
    throughout (see evaluate_real_recordings.py).
  - real_recordings/positives: real media clips of dramatized distress
    (not scripted UDK phrases). Ground truth: at least one event fires
    somewhere in the file.
  - real_recordings/negatives: real ordinary podcast/talk/media audio.
    Ground truth: NO event fires anywhere in the file. Reported both
    per-file (operationally what matters: did this recording ever
    trigger) and per-segment (comparable to this project's earlier
    BEATs/SER false-positive-rate methodology).

Every clip is processed through ALL of its VAD segments with a single
persistent UDKEngine per file/clip, exactly mirroring api.py's real
per-journey behavior (repetition/distinct-UDK escalation state carries
across segments within one file) -- not the older single-longest-segment
simplification some earlier scripts used for single-utterance corpora.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Real, previously-documented bug (see Part 2 of the project doc): the
# Windows console's default cp1252 encoding can't print emoji/non-Latin
# characters that show up in real media filenames (e.g. real_recordings/
# positives|negatives), crashing mid-run. Forcing UTF-8 on stdout fixes
# it at the source instead of working around it ad hoc per script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import librosa
import numpy as np

from audio_augment import add_noise, muffle, overlap, pitch_shift, time_stretch
from evaluate_pipeline_corpus import NEGATIVES, VOICES, _load_pcm_16k_mono, _synthesize_wav
from kws import Wav2Vec2DTWSpotter
from stt import FasterWhisperSTT
from stt_confidence_gate import is_suspect
from udk_engine import UDKEngine
from udks import GENERAL_UDKS
from vad import VAD, SAMPLE_RATE

REAL_RECORDINGS = Path(__file__).parent / "real_recordings"


def default_device() -> str:
    """cuda when a GPU is present, else cpu -- so a Colab run uses the GPU
    even if the env vars below weren't passed (a local CPU box is unchanged)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_real_pcm(path: Path) -> bytes:
    audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


def run_file(pcm: bytes, vad: VAD, stt, kws, semantic_matcher, beats, separator=None):
    """Runs one whole audio file/clip through every VAD segment with one
    persistent engine, mirroring api.py's real per-journey loop. Returns
    (events, per_segment_rows, total_latency_s)."""
    t0 = time.monotonic()
    engine = UDKEngine(GENERAL_UDKS, semantic_matcher=semantic_matcher)
    segments = vad.segment_speech(pcm)
    events = []
    rows = []
    for seg in segments:
        transcript = stt.transcribe(seg.pcm)
        kws_match = kws.spot(seg.pcm) if kws is not None else None
        if not transcript.text and kws_match is None:
            continue
        scream_score = beats.score(seg.pcm) if beats is not None else None
        stt_suspect = is_suspect(transcript)
        pre_state = engine.snapshot()
        decision, *sentence_extras = engine.decide_all(
            transcript.text, now_s=time.monotonic(), kws_match=kws_match,
            scream_score=scream_score, stt_suspect=stt_suspect,
        )
        if sentence_extras:
            if separator is not None and getattr(separator, "recheck_verify", False):
                from separation import separated_extra_event

                extra = separated_extra_event(
                    seg.pcm, decision, sentence_extras, stt, engine, separator, time.monotonic(), kws=kws
                )
                if extra is not None and extra.decision != "NO_ACTION":
                    extra.layer += "+wesep"
                    sentence_extras.append(extra)
        elif decision.decision == "NO_ACTION" and separator is not None:
            from separation import retry_with_separation

            recovered = retry_with_separation(seg.pcm, stt, engine, separator, time.monotonic(), kws=kws)
            if recovered is not None:
                decision = recovered
        elif separator is not None and getattr(separator, "recheck_verify", False):
            from separation import recheck_verify_with_separation

            decision = recheck_verify_with_separation(
                seg.pcm, decision, pre_state, stt, engine, separator, time.monotonic(), kws=kws
            )
        rows.append({
            "transcript": transcript.text, "decision": decision.decision, "start_ms": seg.start_ms, "end_ms": seg.end_ms,
            "udk_id": decision.udk.udk_id if decision.udk else None,
            "layer": decision.layer, "confidence": decision.confidence,
            "scream_score": scream_score, "stt_suspect": stt_suspect,
            # raw KWS match even when a text layer decided -- to calibrate KWS-alone on real audio
            "kws": (kws_match.phrase_id, round(kws_match.distance, 3)) if kws_match is not None else None,
            "_event": decision,  # for apply_intent_gate
        })
        if decision.decision != "NO_ACTION":
            events.append(decision)
        for extra in sentence_extras:
            extra.layer += "/sentence"  # so printed events show it came from the per-sentence pass
            rows.append({
                "transcript": extra.transcript, "decision": extra.decision, "start_ms": seg.start_ms, "end_ms": seg.end_ms,
                "udk_id": extra.udk.udk_id if extra.udk else None,
                "layer": extra.layer, "confidence": extra.confidence,
                "scream_score": scream_score, "stt_suspect": stt_suspect, "_event": extra,
            })
            events.append(extra)
    latency = time.monotonic() - t0
    return events, rows, latency


class _Memo:
    """obj.method memoised by the audio bytes, for one clip. Lets a second STT
    path reuse the first path's VAD segments, KWS and BEATs results. Sharing
    VAD output also matters for correctness: webrtcvad adapts across calls,
    so segmenting the same clip twice can return different segments."""

    def __init__(self, obj, method: str):
        self._fn, self._cache = getattr(obj, method), {}
        setattr(self, method, self)
        if hasattr(obj, "total_ms"):
            self.total_ms = obj.total_ms

    def __call__(self, pcm: bytes, *args, **kwargs):
        if pcm not in self._cache:
            self._cache[pcm] = self._fn(pcm, *args, **kwargs)
        return self._cache[pcm]


def run_file_either(pcm: bytes, vad: VAD, stts: dict, kws, semantic_matcher, beats, separator=None, gate=None) -> dict:
    """run_file() once per STT backend in `stts` ({name: backend}), on the SAME
    VAD segments, reusing KWS/BEATs results across paths. Each path gets its
    own engine, exactly like a Whisper-only run. The clip alerts if EITHER
    path alerts (callers take the stronger tier) -- see stt.ParakeetSTT for
    the measured recall gain. Returns {name: (events, rows, latency_s)}."""
    shared_vad = _Memo(vad, "segment_speech")
    shared_kws = _Memo(kws, "spot") if kws is not None else None
    shared_beats = _Memo(beats, "score") if beats is not None else None
    out = {name: run_file(pcm, shared_vad, stt, shared_kws, semantic_matcher, shared_beats, separator)
           for name, stt in stts.items()}
    if gate is not None and len(out) > 1:
        apply_intent_gate(out, gate)
    return out


def apply_intent_gate(out: dict, gate) -> None:
    """intent_gate.IntentGate on run_file_either's result, in place: an alert whose words
    the OTHER STT also heard (same segment) and the model scores below its threshold is
    dropped from events; its row turns NO_ACTION and keeps the score as "veto_p".
    ponytail: the path's engine already counted the vetoed alert toward repetition
    escalation; replay the engine without it if that ever matters."""
    import intent_gate

    heard, ends = {name: {} for name in out}, {name: {} for name in out}
    for name, (_, rows, _) in out.items():
        for r in rows:
            heard[name].setdefault(r["start_ms"], r["transcript"])  # first row = the whole segment
            ends[name].setdefault(r["start_ms"], r.get("end_ms", r["start_ms"]))
    ctx_ms = getattr(gate, "context_s", 0) * 1000
    for name, (events, rows, _) in out.items():
        for r in rows:
            e = r.get("_event")
            if e is None or r["decision"] == "NO_ACTION":
                continue
            other = " ".join(heard[o].get(r["start_ms"], "") for o in out if o != name)
            earlier = [t for s, t in sorted(heard[name].items())
                       if ctx_ms and s < r["start_ms"] and ends[name][s] >= r["start_ms"] - ctx_ms]
            p = gate.veto(e, other, intent_gate.context_before(heard[name][r["start_ms"]], e.transcript, earlier))
            if p is not None:
                events[:] = [x for x in events if x is not e]
                r["decision"], r["veto_p"] = "NO_ACTION", round(p, 3)


def lat_stats(latencies: list[float]) -> str:
    if not latencies:
        return "n=0"
    arr = sorted(latencies)
    p95 = arr[min(len(arr) - 1, int(0.95 * len(arr)))]
    return (f"n={len(arr)} mean={statistics.mean(arr):.3f}s median={statistics.median(arr):.3f}s "
            f"p95={p95:.3f}s max={max(arr):.3f}s")


def main() -> None:
    print("Loading VAD, real FasterWhisperSTT, real Wav2Vec2 KWS, real semantic matcher, real BEATs...")
    vad = VAD()
    # UDK_EVAL_STT_MODEL: e.g. small.en. UDK_EVAL_DEVICE=cuda puts EVERY model
    # (Whisper, KWS wav2vec2, semantic MiniLM, BEATs, WeSep) on the GPU;
    # UDK_EVAL_STT_DEVICE is the older Whisper-only switch, kept as a fallback.
    device = os.environ.get("UDK_EVAL_DEVICE", os.environ.get("UDK_EVAL_STT_DEVICE", default_device()))
    stt_model = os.environ.get("UDK_EVAL_STT_MODEL", "tiny.en")
    print(f"Device for all models: {device}   STT: faster-whisper {stt_model}")
    stt = FasterWhisperSTT(model_size=stt_model, device=device,
                           compute_type="float16" if device == "cuda" else "int8")
    kws = None
    if os.environ.get("UDK_ENABLE_KWS", "1") != "0":  # UDK_ENABLE_KWS=0: the deployed config (KWS off)
        kws = Wav2Vec2DTWSpotter(device=device)
        kws.load_references(Path(__file__).parent / "kws_references.npz")
    print(f"keyword spotting: {'on' if kws else 'OFF'}")
    from semantic import SentenceTransformerSemanticMatcher

    semantic_matcher = SentenceTransformerSemanticMatcher(device=device)
    from beats_distress_detector import BEATsDistressDetector

    beats = BEATsDistressDetector(device=device)

    # UDK_EVAL_PARAKEET=0: Whisper only (the pre-2026-09-24 config). Default:
    # Whisper + Parakeet on the same segments, a clip alerts if EITHER alerts
    # (run_file_either; same as evaluate_prepped_data.py and api.py).
    stts = {"whisper": stt}
    if os.environ.get("UDK_EVAL_PARAKEET", "1") != "0":
        from stt import ParakeetSTT

        stts["parakeet"] = ParakeetSTT(device=device)
        print(f"+ Parakeet (alert if either), running on: {stts['parakeet'].providers[0]}")
    import intent_gate

    gate = intent_gate.from_env(device)  # UDK_INTENT_GATE=<mDeBERTa dir>: veto-agree on text alerts

    def detect(pcm: bytes, sep):
        """(events, rows, latency) pooled over every STT path: the clip's tier is
        the stronger path's, its UDKs the union. Rows carry start_ms + stt."""
        t0 = time.monotonic()
        runs = run_file_either(pcm, vad, stts, kws, semantic_matcher, beats, sep, gate=gate)
        events = [e for path_events, _, _ in runs.values() for e in path_events]
        rows = [{**r, "stt": name} for name, (_, path_rows, _) in runs.items() for r in path_rows]
        for r in rows:  # printed just above the clip's result line
            if "veto_p" in r:
                print(f"      [intent veto {r['udk_id']} {r['layer']} p={r['veto_p']}] {r['stt']}: {r['transcript'][:100]!r}")
        return events, rows, time.monotonic() - t0

    separator = None
    if os.environ.get("UDK_ENABLE_SEPARATION") == "1":
        print("Loading SepFormer NO_ACTION fallback (UDK_ENABLE_SEPARATION=1)...")
        from separation import SepformerSeparator

        separator = SepformerSeparator()

    # UDK_EVAL_WESEP=1: WeSep target-speaker fallback + VERIFY re-check on
    # the SYNTHETIC corpus only, enrolled as each clip's own TTS voice.
    # Real recordings have no voice sample of their speaker, so they run
    # without it (unchanged path) -- stated in the output, not hidden.
    wesep_by_voice = {}
    if os.environ.get("UDK_EVAL_WESEP") == "1":
        print("Loading WeSep + ECAPA (UDK_EVAL_WESEP=1; synthetic corpus only)...")
        from wesep_extraction import EcapaVerifier, WeSepExtractor, WeSepTargetSeparator

        extractor, verifier = WeSepExtractor(device=device), EcapaVerifier(device=device)
        with tempfile.TemporaryDirectory() as tmp:
            for v in VOICES:
                p = Path(tmp) / "enroll.wav"
                _synthesize_wav(GENERAL_UDKS[0].phrase, v, p)
                wesep_by_voice[v] = WeSepTargetSeparator(_load_pcm_16k_mono(p), extractor=extractor, verifier=verifier)

    def synthetic_separator(i: int):
        return wesep_by_voice.get(VOICES[i % len(VOICES)], separator)

    all_summaries = {}

    # =====================================================================
    # 1. SYNTHETIC CORPUS (evaluate_pipeline_corpus.py's own builder)
    # =====================================================================
    # UDK_EVAL_SKIP_SYNTHETIC=1: the synthetic corpus needs Windows SAPI TTS
    # (PowerShell), so it can't run on Linux/Colab -- real recordings still do.
    import shutil

    if os.environ.get("UDK_EVAL_SKIP_SYNTHETIC") == "1" or shutil.which("powershell") is None:
        why = "UDK_EVAL_SKIP_SYNTHETIC=1" if os.environ.get("UDK_EVAL_SKIP_SYNTHETIC") == "1" else \
            "no PowerShell here -- the synthetic corpus needs Windows SAPI TTS"
        print(f"\n(synthetic corpus skipped: {why})")
    else:
        print(f"\n{'=' * 78}\n1. SYNTHETIC CORPUS (TTS + signal augmentation)\n{'=' * 78}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            print(f"Synthesizing {len(GENERAL_UDKS)} base positive + {len(NEGATIVES)} base negative clips...")
            base_positive: dict[str, bytes] = {}
            for i, udk in enumerate(GENERAL_UDKS):
                wav_path = tmp_path / f"pos_{udk.udk_id}.wav"
                _synthesize_wav(udk.phrase, VOICES[i % len(VOICES)], wav_path)
                base_positive[udk.udk_id] = _load_pcm_16k_mono(wav_path)
            base_negative: list[bytes] = []
            for i, text in enumerate(NEGATIVES):
                wav_path = tmp_path / f"neg_{i}.wav"
                _synthesize_wav(text, VOICES[i % len(VOICES)], wav_path)
                base_negative.append(_load_pcm_16k_mono(wav_path))

            syn_pos_rows, syn_neg_rows = [], []
            conditions = ["baseline", "pitch_tempo_distorted", "muffled", "noisy", "overlapping"]
            print("Running positives across all 5 conditions x 20 UDKs...")
            for i, udk in enumerate(GENERAL_UDKS):
                base = base_positive[udk.udk_id]
                clips = {
                    "baseline": base,
                    "pitch_tempo_distorted": time_stretch(pitch_shift(base, SAMPLE_RATE, n_steps=3), rate=1.3),
                    "muffled": muffle(base, SAMPLE_RATE),
                    "noisy": add_noise(base, SAMPLE_RATE, snr_db=10, kind="traffic"),
                    "overlapping": overlap(base, base_negative[i % len(base_negative)]),
                }
                for cond, pcm in clips.items():
                    events, rows, latency = detect(pcm, synthetic_separator(i))
                    fired_correct = any(e.udk.udk_id == udk.udk_id for e in events)
                    syn_pos_rows.append({"udk_id": udk.udk_id, "condition": cond, "correct": fired_correct, "latency": latency, "n_events": len(events)})

            print("Running negatives (baseline + degraded)...")
            for i, text in enumerate(NEGATIVES):
                base = base_negative[i]
                degraded = add_noise(muffle(base, SAMPLE_RATE), SAMPLE_RATE, snr_db=10, kind="crowd")
                for cond, pcm in [("baseline", base), ("degraded", degraded)]:
                    events, rows, latency = detect(pcm, synthetic_separator(i))
                    syn_neg_rows.append({"text": text, "condition": cond, "false_positive": len(events) > 0, "latency": latency})

        print(f"\n{'condition':<22}{'recall':>10}{'n':>6}{'avg_latency':>14}")
        for cond in conditions:
            rows = [r for r in syn_pos_rows if r["condition"] == cond]
            correct = sum(1 for r in rows if r["correct"])
            avg_lat = statistics.mean(r["latency"] for r in rows)
            print(f"{cond:<22}{correct / len(rows):>10.1%}{len(rows):>6}{avg_lat:>14.3f}s")
        for cond in ["baseline", "degraded"]:
            rows = [r for r in syn_neg_rows if r["condition"] == cond]
            fpr = sum(1 for r in rows if r["false_positive"]) / len(rows)
            avg_lat = statistics.mean(r["latency"] for r in rows)
            print(f"neg_{cond:<18}{fpr:>10.1%}{len(rows):>6}{avg_lat:>14.3f}s")

        syn_correct = sum(1 for r in syn_pos_rows if r["correct"])
        syn_fp = sum(1 for r in syn_neg_rows if r["false_positive"])
        syn_fn = len(syn_pos_rows) - syn_correct
        syn_lat = [r["latency"] for r in syn_pos_rows] + [r["latency"] for r in syn_neg_rows]
        all_summaries["synthetic"] = {"n": len(syn_pos_rows) + len(syn_neg_rows), "correct": syn_correct,
                                       "fp": syn_fp, "fn": syn_fn, "n_pos": len(syn_pos_rows), "n_neg": len(syn_neg_rows),
                                       "latency": syn_lat}

    # =====================================================================
    # 2. real_recordings/mixed (manifest-driven)
    # =====================================================================
    print(f"\n{'=' * 78}\n2. real_recordings/mixed (manifest-driven degraded-condition corpus)\n{'=' * 78}")
    import json

    manifest_path = REAL_RECORDINGS / "mixed" / "manifest.json"
    mixed_rows = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        udk_phrases_lower = {u.phrase.lower(): u.udk_id for u in GENERAL_UDKS}
        for entry in manifest:
            stem = Path(entry["file"]).stem
            candidates = list((REAL_RECORDINGS / "mixed").glob(f"{stem}.*"))
            candidates = [c for c in candidates if c.suffix.lower() in (".mp3", ".wav", ".m4a")]
            if not candidates:
                print(f"  [SKIP] {entry['file']}: no audio file found on disk for stem {stem!r}")
                continue
            path = candidates[0]
            try:
                pcm = load_real_pcm(path)
            except Exception as exc:
                print(f"  [SKIP] {path.name}: failed to load ({exc})")
                continue
            expected_udk = udk_phrases_lower.get(entry["phrase"].lower())
            events, rows, latency = detect(pcm, separator)
            detected = len(events) > 0
            exact_ok = expected_udk is not None and any(e.udk.udk_id == expected_udk for e in events)
            correct = exact_ok if expected_udk is not None else detected
            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] {entry['file']:<28} cond={entry['condition']:<12} phrase={entry['phrase']!r}")
            top = events[0] if events else None
            print(f"         detected={detected} matched={top.udk.udk_id if top else None} "
                  f"decision={top.decision if top else 'NO_ACTION'} latency={latency:.3f}s")
            mixed_rows.append({"file": entry["file"], "condition": entry["condition"], "correct": correct,
                                "detected": detected, "latency": latency, "had_exact_udk": expected_udk is not None})
    else:
        print(f"  (no manifest found at {manifest_path})")

    print(f"\n{'condition':<15}{'recall':>10}{'n':>6}{'avg_latency':>14}")
    for cond in sorted(set(r["condition"] for r in mixed_rows)):
        rows = [r for r in mixed_rows if r["condition"] == cond]
        correct = sum(1 for r in rows if r["correct"])
        avg_lat = statistics.mean(r["latency"] for r in rows)
        print(f"{cond:<15}{correct / len(rows):>10.1%}{len(rows):>6}{avg_lat:>14.3f}s")

    mixed_correct = sum(1 for r in mixed_rows if r["correct"])
    all_summaries["mixed"] = {"n": len(mixed_rows), "correct": mixed_correct, "fp": 0,
                               "fn": len(mixed_rows) - mixed_correct, "n_pos": len(mixed_rows), "n_neg": 0,
                               "latency": [r["latency"] for r in mixed_rows]}

    # =====================================================================
    # 3. real_recordings/positives (real media, dramatized distress)
    # =====================================================================
    print(f"\n{'=' * 78}\n3. real_recordings/positives (real media clips)\n{'=' * 78}")
    pos_dir = REAL_RECORDINGS / "positives"
    pos_rows = []
    for path in sorted(pos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".mp3", ".wav", ".m4a"):
            continue
        try:
            pcm = load_real_pcm(path)
        except Exception as exc:
            print(f"  [SKIP] {path.name}: failed to load ({exc})")
            continue
        events, rows, latency = detect(pcm, separator)
        detected = len(events) > 0
        status = "PASS" if detected else "FAIL (missed)"
        print(f"  [{status}] {path.name[:60]:<60} events={len(events)} latency={latency:.3f}s")
        for e in events:
            print(f"         -> {e.decision} udk={e.udk.udk_id if e.udk else None} conf={e.confidence:.2f} layer={e.layer}")
        pos_rows.append({"file": path.name, "detected": detected, "latency": latency, "n_events": len(events)})

    pos_correct = sum(1 for r in pos_rows if r["detected"])
    print(f"\nReal-media positive recall: {pos_correct}/{len(pos_rows)} ({pos_correct / len(pos_rows):.1%})" if pos_rows else "\n(no positive files found)")
    all_summaries["real_positives"] = {"n": len(pos_rows), "correct": pos_correct, "fp": 0,
                                        "fn": len(pos_rows) - pos_correct, "n_pos": len(pos_rows), "n_neg": 0,
                                        "latency": [r["latency"] for r in pos_rows]}

    # =====================================================================
    # 4. real_recordings/negatives (real ordinary media)
    # =====================================================================
    print(f"\n{'=' * 78}\n4. real_recordings/negatives (real ordinary media clips)\n{'=' * 78}")
    neg_dir = REAL_RECORDINGS / "negatives"
    neg_rows = []
    total_segments = 0
    total_segment_fps = 0
    for path in sorted(neg_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".mp3", ".wav", ".m4a"):
            continue
        try:
            pcm = load_real_pcm(path)
        except Exception as exc:
            print(f"  [SKIP] {path.name}: failed to load ({exc})")
            continue
        events, rows, latency = detect(pcm, separator)
        # per SEGMENT (start_ms), counted once across STT paths; a segment is a false
        # positive if EITHER path fired on it
        total_segments += len({r["start_ms"] for r in rows})
        total_segment_fps += len({r["start_ms"] for r in rows if r["decision"] != "NO_ACTION"})
        false_positive = len(events) > 0
        status = "FALSE POSITIVE" if false_positive else "correct (no trigger)"
        print(f"  [{status}] {path.name[:60]:<60} segments={len({r['start_ms'] for r in rows})} events={len(events)} latency={latency:.3f}s")
        for e in events:
            print(f"         -> {e.decision} udk={e.udk.udk_id if e.udk else None} conf={e.confidence:.2f} layer={e.layer} transcript={e.transcript!r}")
        neg_rows.append({"file": path.name, "false_positive": false_positive, "latency": latency, "n_segments": len(rows)})

    neg_fp = sum(1 for r in neg_rows if r["false_positive"])
    print(f"\nReal-media per-FILE false-positive rate: {neg_fp}/{len(neg_rows)} ({neg_fp / len(neg_rows):.1%})" if neg_rows else "")
    print(f"Real-media per-SEGMENT false-positive rate: {total_segment_fps}/{total_segments} "
          f"({total_segment_fps / total_segments:.1%})" if total_segments else "")
    all_summaries["real_negatives"] = {"n": len(neg_rows), "correct": len(neg_rows) - neg_fp, "fp": neg_fp,
                                        "fn": 0, "n_pos": 0, "n_neg": len(neg_rows),
                                        "latency": [r["latency"] for r in neg_rows]}

    # =====================================================================
    # CONSOLIDATED REPORT
    # =====================================================================
    print(f"\n{'=' * 78}\nCONSOLIDATED REPORT -- ENTIRE SYSTEM\n{'=' * 78}")
    print(f"{'Corpus':<18}{'n':>6}{'correct':>10}{'FP':>6}{'FN':>6}{'latency':>50}")
    total_n = total_correct = total_fp = total_fn = 0
    all_latencies = []
    for name, s in all_summaries.items():
        print(f"{name:<18}{s['n']:>6}{s['correct']:>10}{s['fp']:>6}{s['fn']:>6}   {lat_stats(s['latency'])}")
        total_n += s["n"]
        total_correct += s["correct"]
        total_fp += s["fp"]
        total_fn += s["fn"]
        all_latencies.extend(s["latency"])
    print(f"{'-' * 78}")
    print(f"{'TOTAL':<18}{total_n:>6}{total_correct:>10}{total_fp:>6}{total_fn:>6}   {lat_stats(all_latencies)}")
    print(f"\nOverall accuracy (correct / n): {total_correct / total_n:.1%}" if total_n else "")
    print(f"Overall false-positive rate (FP / total negatives tested): "
          f"{total_fp / sum(s['n_neg'] for s in all_summaries.values()):.1%}"
          if sum(s["n_neg"] for s in all_summaries.values()) else "")
    print(f"Overall false-negative rate (FN / total positives tested): "
          f"{total_fn / sum(s['n_pos'] for s in all_summaries.values()):.1%}"
          if sum(s["n_pos"] for s in all_summaries.values()) else "")


if __name__ == "__main__":
    main()
