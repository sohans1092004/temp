"""Full-corpus validation of stt_confidence_gate.py + dual_asr_guard.py,
per the conversation's request: this session's validation was targeted
at known cases (clip_08 suppression, clip_01 true-positive regression) --
this runs the entire real_recordings/ corpus (positives+negatives+mixed,
n=60) plus the 20 TTS-synthesized adversarial negatives already built
this session, across 6 pipeline passes, to get real aggregate numbers.

Loads every heavy backend ONCE (api.py's module-level singletons), then
toggles api._stt_confidence_gate_enabled / api._dual_asr_guard directly
between passes -- avoids reloading BEATs/wav2vec2/faster-whisper 6 times.

Captures every UDKEngine.decide() call (not just ones that fire an
event) via a trace wrapper, since a vetoed match never reaches
journey.events -- the only way to see suppression happening is to watch
every decide() call's reason string.
"""

from __future__ import annotations

import json
import os
import uuid

os.environ.setdefault("UDK_ENABLE_STT", "1")
os.environ.setdefault("UDK_ENABLE_SCREAM_DETECTION", "1")
os.environ.setdefault("UDK_ENABLE_DUAL_ASR", "1")
os.environ.setdefault("UDK_ENABLE_STT_CONFIDENCE_GATE", "1")

import librosa
import numpy as np

import api
from udk_engine import UDKEngine

SAMPLE_RATE = 16_000
CHUNK_BYTES = int(SAMPLE_RATE * 0.5) * 2

MIXED_DIR = "real_recordings/mixed"
POSITIVES_DIR = "real_recordings/positives"
NEGATIVES_DIR = "real_recordings/negatives"

with open(os.path.join(MIXED_DIR, "manifest.json"), encoding="utf-8") as f:
    MANIFEST = {os.path.splitext(e["file"])[0]: e for e in json.load(f)}

EXACT_UDK_BY_STEM = {
    "clip_01_clean": "UDK_03", "clip_02_noisy": "UDK_04", "clip_03_muffled": "UDK_05",
    "clip_04_overlapping": "UDK_06", "clip_05_clean": "UDK_07", "clip_06_noisy": "UDK_08",
    "clip_07_muffled": "UDK_09", "clip_08_overlapping": "UDK_10", "clip_09_clean": "UDK_11",
    "clip_10_noisy": "UDK_12", "clip_11_muffled": "UDK_13", "clip_12_noisy": "UDK_14",
    "clip_13_clean": "UDK_15", "clip_14_noisy": "UDK_16", "clip_15_muffled": "UDK_17",
    "clip_16_overlapping": "UDK_18", "clip_17_muffled": "UDK_19", "clip_18_noisy": "UDK_20",
}

# The same 20 adversarial sentences built this session in
# Model-snr/validate_forced_scoring.py -- UDK words embedded in neutral,
# non-distress contexts. These target the class that broke forced-
# phrase-scoring (95% FP rate); neither guardrail here was designed for
# it (the STT gets these right, correctly -- it's a semantic false
# positive, not a hallucination or a transcript disagreement).
ADVERSARIAL_SENTENCES = {
    "Call the police, I need help": "I'll call the police department later, I need help filling out this form",
    "I'm not safe": "This old ladder is not safe, we should really buy a new one",
    "Get away from me": "Get away from me, you are blocking the TV",
    "Someone is following me": "Someone is following me on Instagram now, that's so nice of them",
    "I need help right now": "I need help right now finding my car keys",
    "Don't touch me": "Do not touch me, I just got a manicure and it's still wet",
    "I'm being followed": "I'm being followed by three cats every single morning, it's adorable",
    "Help me, please": "Help me, please pass the salt from your side of the table",
    "I'm scared, stay back": "I'm scared of heights, so stay back from the edge for this photo",
    "Let go of me": "Can you let go of the door handle, it seems to be stuck",
    "I need the police here": "I need the police station's address, is it right here on this street",
    "Someone is trying to hurt me": "Someone is trying to hurt my feelings with this joke, ha ha, very funny",
    "Please don't hurt me": "Please don't hurt yourself moving that heavy couch alone",
    "I don't feel safe here": "I don't feel safe here financially, we should really save more this year",
    "Stay away from me": "Stay away from me for a few days, I have a cold and don't want you to catch it",
    "I'm in danger": "I'm in danger of eating this entire pizza all by myself tonight",
    "Somebody help me now": "Somebody help me move this couch right now, it's really heavy",
    "I want to go home now": "I want to go home now, the movie was great but I'm exhausted",
    "Please call for help": "Please call for help with the printer, it's jammed again",
    "I need to get out of here": "I need to get out of here early tomorrow for my dentist appointment",
}

TRACE: list[dict] = []
DUAL_ASR_CALL_LOG: list[str] = []  # one entry per real disagrees() invocation -- the trigger firing


def _install_dual_asr_call_tracker(guard) -> None:
    """Wraps disagrees() to log every invocation -- the only reliable way
    to know the temperature==0.0 trigger condition actually fired,
    independent of whether it agreed or disagreed."""
    orig_disagrees = guard.disagrees

    def tracked(primary_transcript, pcm, sample_rate=16000):
        DUAL_ASR_CALL_LOG.append(primary_transcript)
        return orig_disagrees(primary_transcript, pcm, sample_rate)

    guard.disagrees = tracked


def _install_trace() -> None:
    orig_decide = UDKEngine.decide

    def traced(self, transcript, now_s=None, kws_match=None, scream_score=None, stt_suspect=False, dual_asr_disagreement=False):
        result = orig_decide(
            self, transcript, now_s=now_s, kws_match=kws_match, scream_score=scream_score,
            stt_suspect=stt_suspect, dual_asr_disagreement=dual_asr_disagreement,
        )
        TRACE.append({
            "transcript": transcript,
            "decision": result.decision,
            "udk": result.udk.udk_id if result.udk else None,
            "reason": result.reason,
            "stt_suspect": stt_suspect,
            "dual_asr_disagreement": dual_asr_disagreement,
        })
        return result

    UDKEngine.decide = traced


_install_trace()


def load_pcm(path: str, sr: int = SAMPLE_RATE) -> bytes:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


def run_clip(pcm: bytes, label: str) -> dict:
    """Runs one clip through the real production pipeline exactly like
    run_real_audio.py, returns fired events + the decide() trace for
    just this clip."""
    start = len(TRACE)
    dual_asr_calls_start = len(DUAL_ASR_CALL_LOG)
    journey_id = f"J_validate_{label}_{uuid.uuid4().hex[:8]}"
    journey, _ = api._store.create_or_get(
        journey_id, "U_validate", api._stt_backend, api.SAMPLE_PERSONAL_UDK, api._audio_store,
        api._kws_backend, api._semantic_matcher,
        multilingual_semantic_matcher=api._multilingual_semantic_matcher,
        stt_by_language=api._indic_stt_backends, kws_by_language=api._indic_kws_backends,
        language_detector=None, language_recheck_detector=None,
    )
    seq = 0
    for i in range(0, len(pcm), CHUNK_BYTES):
        api._ingest_frame(journey, seq, pcm[i:i + CHUNK_BYTES], api._audio_store)
        seq += 1
    api._run_incremental_detection(journey, final=True)
    clip_trace = TRACE[start:]
    return {
        "fired": len(journey.events) > 0,
        "udks_fired": [e["udk_id"] for e in journey.events],
        "trace": clip_trace,
        "vetoed_by_gate": sum(1 for t in clip_trace if "STT-confidence guardrail" in (t["reason"] or "")),
        "vetoed_by_dual_asr": sum(1 for t in clip_trace if "dual-ASR disagreement" in (t["reason"] or "")),
        "dual_asr_invocations": len(DUAL_ASR_CALL_LOG) - dual_asr_calls_start,
    }


def set_config(gate_on: bool, dual_asr_on: bool) -> None:
    api._stt_confidence_gate_enabled = gate_on
    api._dual_asr_guard = _REAL_DUAL_ASR_GUARD if dual_asr_on else None


def collect_corpus() -> dict[str, list[tuple[str, str]]]:
    """Returns {category: [(label, path), ...]}."""
    corpus = {"positives": [], "negatives": [], "mixed_exact": [], "mixed_paraphrase": []}
    for f in sorted(os.listdir(POSITIVES_DIR)):
        p = os.path.join(POSITIVES_DIR, f)
        if os.path.isfile(p):
            corpus["positives"].append((f, p))
    for f in sorted(os.listdir(NEGATIVES_DIR)):
        p = os.path.join(NEGATIVES_DIR, f)
        if os.path.isfile(p):
            corpus["negatives"].append((f, p))
    for f in sorted(os.listdir(MIXED_DIR)):
        if not f.endswith(".mp3"):
            continue
        stem = os.path.splitext(f)[0]
        p = os.path.join(MIXED_DIR, f)
        if stem in EXACT_UDK_BY_STEM:
            corpus["mixed_exact"].append((f, p))
        else:
            corpus["mixed_paraphrase"].append((f, p))
    return corpus


def run_full_pass(corpus: dict, pass_name: str) -> dict:
    print(f"\n{'=' * 70}\nPASS: {pass_name}\n{'=' * 70}")
    results = {}
    for category, items in corpus.items():
        cat_results = []
        for label, path in items:
            try:
                pcm = load_pcm(path)
            except Exception as e:
                print(f"  [SKIP] {label}: {e}")
                continue
            r = run_clip(pcm, f"{pass_name}_{category}_{label}")
            cat_results.append((label, r))
        results[category] = cat_results
        n = len(cat_results)
        n_fired = sum(1 for _, r in cat_results if r["fired"])
        print(f"  {category:18s} n={n:3d}  fired={n_fired:3d} ({100*n_fired/n:.0f}%)" if n else f"  {category}: n=0")
    return results


def summarize_exact_correctness(exact_results: list[tuple[str, dict]]) -> tuple[int, int]:
    n_correct = 0
    for label, r in exact_results:
        stem = os.path.splitext(label)[0]
        expected = EXACT_UDK_BY_STEM.get(stem)
        if expected and expected in r["udks_fired"]:
            n_correct += 1
    return n_correct, len(exact_results)


def main() -> None:
    global _REAL_DUAL_ASR_GUARD
    _REAL_DUAL_ASR_GUARD = api._dual_asr_guard
    assert _REAL_DUAL_ASR_GUARD is not None, "dual-ASR guard must load for this validation"
    _install_dual_asr_call_tracker(_REAL_DUAL_ASR_GUARD)

    corpus = collect_corpus()
    for cat, items in corpus.items():
        print(f"corpus[{cat}] = {len(items)} clips")

    print(f"\nTotal real clips: {sum(len(v) for v in corpus.values())}")

    all_results = {}

    print("\n\n########## PASS 1/6: BASELINE (both guardrails OFF) ##########")
    set_config(gate_on=False, dual_asr_on=False)
    all_results["baseline"] = run_full_pass(corpus, "baseline")

    print("\n\n########## PASS 2/6: BOTH guardrails ON ##########")
    set_config(gate_on=True, dual_asr_on=True)
    all_results["both_on"] = run_full_pass(corpus, "both_on")

    print("\n\n########## PASS 3/6: GATE ONLY (dual-ASR off) ##########")
    set_config(gate_on=True, dual_asr_on=False)
    all_results["gate_only"] = run_full_pass(corpus, "gate_only")

    print("\n\n########## PASS 4/6: DUAL-ASR ONLY (run 1/3, gate off) ##########")
    set_config(gate_on=False, dual_asr_on=True)
    all_results["dual_asr_only_run1"] = run_full_pass(corpus, "dual_asr_run1")

    print("\n\n########## PASS 5/6: DUAL-ASR ONLY (run 2/3, gate off) ##########")
    all_results["dual_asr_only_run2"] = run_full_pass(corpus, "dual_asr_run2")

    print("\n\n########## PASS 6/6: DUAL-ASR ONLY (run 3/3, gate off) ##########")
    all_results["dual_asr_only_run3"] = run_full_pass(corpus, "dual_asr_run3")

    print("\n\n########## ADVERSARIAL: 20 TTS sentences, BOTH guardrails ON ##########")
    set_config(gate_on=True, dual_asr_on=True)
    from transformers import AutoTokenizer, VitsModel
    import torch

    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    def synth_pcm(text: str) -> bytes:
        inputs = tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        audio = output.squeeze().numpy().astype(np.float32)
        return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()

    adversarial_results = []
    for true_phrase, sentence in ADVERSARIAL_SENTENCES.items():
        pcm = synth_pcm(sentence)
        r = run_clip(pcm, f"adversarial_{true_phrase[:10]}")
        adversarial_results.append((true_phrase, sentence, r))
        marker = "FIRED" if r["fired"] else "ok"
        print(f"  [{marker}] {true_phrase!r:32s} udks_fired={r['udks_fired']}")

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for pass_name in ("baseline", "both_on", "gate_only", "dual_asr_only_run1", "dual_asr_only_run2", "dual_asr_only_run3"):
        res = all_results[pass_name]
        pos_n = len(res["positives"]); pos_fired = sum(1 for _, r in res["positives"] if r["fired"])
        neg_n = len(res["negatives"]); neg_fired = sum(1 for _, r in res["negatives"] if r["fired"])
        mex_n = len(res["mixed_exact"]); mex_fired = sum(1 for _, r in res["mixed_exact"] if r["fired"])
        mex_correct, _ = summarize_exact_correctness(res["mixed_exact"])
        mpar_n = len(res["mixed_paraphrase"]); mpar_fired = sum(1 for _, r in res["mixed_paraphrase"] if r["fired"])
        overlap_items = [(l, r) for l, r in res["mixed_exact"] if "overlapping" in l]
        overlap_correct, overlap_n = summarize_exact_correctness(overlap_items)
        gate_vetoes = sum(r["vetoed_by_gate"] for cat in res.values() for _, r in cat)
        dual_vetoes = sum(r["vetoed_by_dual_asr"] for cat in res.values() for _, r in cat)
        print(f"\n--- {pass_name} ---")
        print(f"  positives (n={pos_n}):  fired {pos_fired}/{pos_n} ({100*pos_fired/pos_n:.0f}%)" if pos_n else "  positives: n=0")
        print(f"  negatives (n={neg_n}):  fired {neg_fired}/{neg_n} ({100*neg_fired/neg_n:.0f}%) [FALSE POSITIVE RATE]" if neg_n else "  negatives: n=0")
        print(f"  mixed exact-phrase (n={mex_n}): any-fire {mex_fired}/{mex_n}, CORRECT UDK {mex_correct}/{mex_n}")
        print(f"    of which overlapping (n={overlap_n}): correct UDK {overlap_correct}/{overlap_n}")
        print(f"  mixed paraphrase (n={mpar_n}): fired {mpar_fired}/{mpar_n} (NO_ACTION is often correct here)")
        print(f"  total gate vetoes: {gate_vetoes}  total dual-ASR vetoes: {dual_vetoes}")

    print("\n--- Adversarial (n=20, both guardrails ON) ---")
    n_fp = sum(1 for _, _, r in adversarial_results if r["fired"])
    print(f"  fired (false positive): {n_fp}/20 ({100*n_fp/20:.0f}%)")
    print("  (expected: unaffected by either guardrail -- these are semantic FPs, not hallucinations/disagreements)")

    print("\n--- Regression check: any genuine positive newly suppressed vs. baseline? ---")
    baseline = all_results["baseline"]
    both_on = all_results["both_on"]
    for category in ("positives", "mixed_exact"):
        base_map = dict(baseline[category])
        both_map = dict(both_on[category])
        for label in base_map:
            if base_map[label]["fired"] and not both_map[label]["fired"]:
                print(f"  *** REGRESSION: {category}/{label} fired at baseline, suppressed with guardrails ON ***")
    print("  (no output above this line in this section = no regressions found)")

    print("\n--- Dual-ASR trigger stability (temperature==0.0 condition firing across 3 identical runs) ---")
    r1, r2, r3 = all_results["dual_asr_only_run1"], all_results["dual_asr_only_run2"], all_results["dual_asr_only_run3"]
    n_total = 0
    n_always = n_never = n_sometimes = 0
    for cat in r1:
        m1, m2, m3 = dict(r1[cat]), dict(r2[cat]), dict(r3[cat])
        for label in m1:
            if label not in m2 or label not in m3:
                continue
            n_total += 1
            fired = [m1[label]["dual_asr_invocations"] > 0, m2[label]["dual_asr_invocations"] > 0, m3[label]["dual_asr_invocations"] > 0]
            if all(fired):
                n_always += 1
            elif not any(fired):
                n_never += 1
            else:
                n_sometimes += 1
                print(f"  INCONSISTENT trigger: {cat}/{label} fired on runs {[i+1 for i,f in enumerate(fired) if f]} of 3")
    print(f"  n={n_total} clips compared across 3 identical runs:")
    print(f"    triggered on all 3 runs:    {n_always}")
    print(f"    triggered on 0 of 3 runs:   {n_never}")
    print(f"    INCONSISTENT (1 or 2 of 3): {n_sometimes}  <- this is the real probabilistic-catch-rate number")


if __name__ == "__main__":
    main()
