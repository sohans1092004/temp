"""Fair overlapping-speech test for WeSep target-speaker extraction.

Why a new corpus: evaluate_full_system.py's "overlapping" condition mixes
each UDK with an interfering clip spoken by the SAME TTS voice (20/20
clips) -- target-speaker extraction cannot separate a voice from itself,
and a real aggressor is a different person. Real recordings have no
enrollment sample of their speaker, so WeSep can't run on them at all.

Setup (same enrolled user + enrollment clip WeSep was validated with,
commits 5611f64/5f8e667): enrolled user = Zira; enrollment = Zira saying
UDK_01's phrase. Every clip goes through evaluate_full_system.run_file
(VAD -> STT tiny.en + KWS -> BEATs -> confidence gate -> UDKEngine), once
with no fallback and once with WeSepTargetSeparator as the NO_ACTION
fallback.

  Positives (want: the right UDK fires)
    same_gender_overlap  Zira says UDK_i, Hazel talks over it (20)
    diff_gender_overlap  Zira says UDK_i, David talks over it (20)
  Negatives (want: nothing fires)
    user_alone           Zira ordinary talk (20)
    user_plus_other      Zira ordinary talk + David ordinary talk (20)
    user_absent          Hazel + David ordinary talk, Zira not there (20)

Every clip's sha256 is saved to wesep_overlap_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_augment import overlap
from evaluate_full_system import run_file
from evaluate_pipeline_corpus import NEGATIVES, _load_pcm_16k_mono, _synthesize_wav
from udks import GENERAL_UDKS

USER, SAME_GENDER, DIFF_GENDER = "Microsoft Zira Desktop", "Microsoft Hazel Desktop", "Microsoft David Desktop"
MANIFEST = Path(__file__).parent / "wesep_overlap_manifest.json"


class LoggingSeparator:
    """Records every WeSep self-check score so accepts/rejects are reported."""

    def __init__(self, inner):
        self.inner, self.scores = inner, []
        # Must forward this flag, or run_file never applies the VERIFY re-check
        # (bug found 2026-09-23: v2 run silently tested NO_ACTION-only WeSep).
        self.recheck_verify = getattr(inner, "recheck_verify", False)

    def separate(self, pcm: bytes) -> list[bytes]:
        out = self.inner.separate(pcm)
        self.scores.append(self.inner.last_score)
        return out


def main() -> None:
    from beats_distress_detector import BEATsDistressDetector
    from kws import Wav2Vec2DTWSpotter
    from semantic import SentenceTransformerSemanticMatcher
    from stt import FasterWhisperSTT
    from vad import VAD
    from wesep_extraction import EXTRACTION_ACCEPT_THRESHOLD, WeSepTargetSeparator

    print("Loading VAD, STT tiny.en, KWS, semantic, BEATs, WeSep + ECAPA...")
    vad = VAD()
    stt = FasterWhisperSTT(model_size="tiny.en", device="cpu", compute_type="int8")
    kws = Wav2Vec2DTWSpotter()
    kws.load_references(Path(__file__).parent / "kws_references.npz")
    semantic, beats = SentenceTransformerSemanticMatcher(), BEATsDistressDetector()

    manifest = {"clips": []}
    with tempfile.TemporaryDirectory() as tmp:
        def synth(text: str, voice: str, name: str) -> bytes:
            p = Path(tmp) / f"{name}.wav"
            _synthesize_wav(text, voice, p)
            return _load_pcm_16k_mono(p)

        def record(name: str, pcm: bytes) -> bytes:
            manifest["clips"].append({"name": name, "sha256": hashlib.sha256(pcm).hexdigest()})
            return pcm

        print("Synthesizing corpus...")
        enroll = record("enrollment", synth(GENERAL_UDKS[0].phrase, USER, "enroll"))
        user_udk = [synth(u.phrase, USER, f"u_udk{i}") for i, u in enumerate(GENERAL_UDKS)]
        user_neg = [synth(t, USER, f"u_neg{i}") for i, t in enumerate(NEGATIVES)]
        sg_neg = [synth(t, SAME_GENDER, f"sg_neg{i}") for i, t in enumerate(NEGATIVES)]
        dg_neg = [synth(t, DIFF_GENDER, f"dg_neg{i}") for i, t in enumerate(NEGATIVES)]
        n = len(GENERAL_UDKS)
        conditions = {
            "same_gender_overlap": [(GENERAL_UDKS[i].udk_id, record(f"sg_ovl{i}", overlap(user_udk[i], sg_neg[i]))) for i in range(n)],
            "diff_gender_overlap": [(GENERAL_UDKS[i].udk_id, record(f"dg_ovl{i}", overlap(user_udk[i], dg_neg[i]))) for i in range(n)],
            "user_alone": [(None, record(f"u_alone{i}", user_neg[i])) for i in range(n)],
            "user_plus_other": [(None, record(f"u_plus{i}", overlap(user_neg[i], dg_neg[(i + 1) % n]))) for i in range(n)],
            "user_absent": [(None, record(f"absent{i}", overlap(sg_neg[i], dg_neg[(i + 1) % n]))) for i in range(n)],
        }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"  {len(manifest['clips'])} clips, checksums -> {MANIFEST.name}")

    wesep = LoggingSeparator(WeSepTargetSeparator(enroll))
    results = {}
    for config, separator in [("baseline", None), ("wesep", wesep)]:
        print(f"\n{'=' * 78}\nCONFIG: {config}\n{'=' * 78}")
        for cond, clips in conditions.items():
            rows = []
            for i, (expected, pcm) in enumerate(clips):
                before = len(wesep.scores)
                events, seg_rows, latency = run_file(pcm, vad, stt, kws, semantic, beats, separator)
                ok = any(e.udk.udk_id == expected for e in events) if expected else not events
                rows.append({"ok": ok, "latency": latency, "wesep_scores": wesep.scores[before:]})
                if not ok or (config == "wesep" and wesep.scores[before:]):
                    what = ", ".join(f"{e.decision}/{e.udk.udk_id}/{e.layer}" for e in events) or "NO_ACTION"
                    sc = " wesep=" + ",".join(f"{s:.3f}" for s in wesep.scores[before:]) if wesep.scores[before:] else ""
                    print(f"  [{'ok ' if ok else 'BAD'}] {cond} #{i:2d} want={expected or 'none'} got={what} "
                          f"transcripts={[r['transcript'] for r in seg_rows]!r}{sc}")
            results[(config, cond)] = rows

    print(f"\n{'=' * 78}\nSUMMARY (WeSep accept threshold {EXTRACTION_ACCEPT_THRESHOLD})\n{'=' * 78}")
    print(f"{'condition':<22}{'kind':<10}{'baseline':>12}{'wesep':>12}{'lat base':>11}{'lat wesep':>11}  wesep accepts")
    for cond, clips in conditions.items():
        kind = "recall" if clips[0][0] else "FPR"
        cells = []
        for config in ("baseline", "wesep"):
            rows = results[(config, cond)]
            good = sum(r["ok"] for r in rows)
            cells.append(f"{good}/{len(rows)}" if kind == "recall" else f"{len(rows) - good}/{len(rows)}")
        lat = [statistics.mean(r["latency"] for r in results[(c, cond)]) for c in ("baseline", "wesep")]
        scores = [s for r in results[("wesep", cond)] for s in r["wesep_scores"]]
        acc = sum(s >= EXTRACTION_ACCEPT_THRESHOLD for s in scores)
        print(f"{cond:<22}{kind:<10}{cells[0]:>12}{cells[1]:>12}{lat[0]:>10.2f}s{lat[1]:>10.2f}s  {acc}/{len(scores)} segments")
    print("\n(recall: right UDK fired / clips; FPR: clips with any event / clips)")


if __name__ == "__main__":
    main()
