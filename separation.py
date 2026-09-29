"""
Speech separation as a NO_ACTION fallback for overlapping speech (Section
2/12's hardest documented case, measured at 50% recall in
evaluate_pipeline_corpus.py -- an altercation/struggle is exactly the
overlapping-speaker scenario this product cares most about).

Real measured tradeoff (10 real overlap failures reproduced from the
corpus, speechbrain/sepformer-wsj02mix pretrained on CPU):
  - 3/10 of those failures recovered when the mixed segment is split into
    two estimated per-speaker streams and each is re-run through
    STT+KWS+UDKEngine -- 50% -> 65% recall on the "overlapping" condition.
  - 1 NEW false positive out of 20 single-speaker negatives run through
    the same fallback (separation artifacts on clean single-speaker audio
    occasionally hallucinate a second "speaker" whose garbled transcript
    clears TRIGGER_VERIFY). Not zero-cost.
  - ~4s of CPU latency per ~2.5s segment, x2 streams re-transcribed --
    real cost, not negligible for a "real-time" system.

Because of that cost (both the new FP path and the latency), this only
ever runs when the primary STT(+KWS) pass already returned NO_ACTION --
never on a clip that already triggered, and never on every clip
unconditionally the way KWS/semantic run alongside every segment. Opt-in
via UDK_ENABLE_SEPARATION=1, same pattern as UDK_ENABLE_KWS/
UDK_ENABLE_SEMANTIC in api.py, for the same reason: torch + speechbrain
shouldn't slow down every existing test/import for callers who don't
need this.

Not a claim this closes the overlapping-speech gap -- 65% is still far
from the 100% baseline gets, and the model is a generic 2-speaker
separator (WSJ0-2mix, read speech) with no exposure to this product's
audio conditions. Kept because a real, measured 15-point recall gain on
the hardest documented condition is worth the fallback's narrow blast
radius; not wired in as an always-on stage."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from udk_engine import UDKEngine

DEFAULT_MODEL = "speechbrain/sepformer-wsj02mix"


class SeparatorBackend(Protocol):
    def separate(self, pcm: bytes) -> list[bytes]: ...


class SepformerSeparator:
    """Real backend. Deferred import, same reasoning as stt.py/kws.py/
    semantic.py -- speechbrain pulls in torch, which a caller who never
    enables UDK_ENABLE_SEPARATION shouldn't have to pay startup cost for."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        self._model = SepformerSeparation.from_hparams(
            source=model_name, savedir="sepformer_cache", local_strategy=LocalStrategy.COPY
        )
        self.model_version = model_name

    def separate(self, pcm: bytes) -> list[bytes]:
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        wav = torch.from_numpy(audio).unsqueeze(0)
        estimated = self._model.separate_batch(wav)  # (1, T, n_speakers)
        streams = []
        for i in range(estimated.shape[-1]):
            stream = estimated[0, :, i].detach().numpy()
            stream = np.clip(stream / (np.abs(stream).max() + 1e-9), -1, 1)
            streams.append((stream * 32767.0).astype("<i2").tobytes())
        return streams


def retry_with_separation(pcm, stt, engine, separator: SeparatorBackend, now_s: float, kws=None):
    """Only called by the caller when the primary STT(+KWS) pass already
    returned NO_ACTION. Re-runs STT(+KWS) on each separated stream and
    returns the highest-confidence non-NO_ACTION decision, or None if
    every stream still misses -- the caller keeps its original NO_ACTION
    event in that case.

    Real bug found and fixed here: an earlier version called
    `engine.decide()` directly on the live engine once per stream. Two
    separated streams are alternate GUESSES about the same single moment,
    not sequential real utterances -- but each non-NO_ACTION guess still
    mutates the engine's repetition/distinct-UDK windows (a NO_ACTION
    guess doesn't, but a wrong-but-above-threshold one does). Measured
    real failure: on one overlapping-negative test clip, two garbled
    separated-stream transcripts each independently cleared 0.60 on
    DIFFERENT UDKs, and the distinct-UDK-escalation rule (built for two
    different REAL phrases said in a row) mistook these two simultaneous
    misreadings of one segment for that, forcing an unconfirmed
    TRIGGER_ALL at only 0.648 confidence. Fixed by probing each stream
    against a throwaway engine (so simultaneous stream guesses can never
    cross-contaminate each other's window state), then applying only the
    single winning candidate to the real engine exactly once -- so
    genuine repeat/distinct tracking across real, separate detections
    over time still works correctly."""
    best = _best_candidate(pcm, stt, engine, separator, now_s, kws)
    if best is None:
        return None
    return engine.decide(best[0], now_s=now_s, kws_match=best[1])


def _best_candidate(pcm, stt, engine, separator, now_s, kws):
    """(transcript, kws_match, probe_decision) of the strongest separated
    stream, scored on throwaway engines -- the live engine is untouched."""
    candidates = []
    for stream_pcm in separator.separate(pcm):
        transcript = stt.transcribe(stream_pcm)
        kws_match = kws.spot(stream_pcm) if kws is not None else None
        if not transcript.text and kws_match is None:
            continue
        probe = UDKEngine(engine.udks, semantic_matcher=engine.semantic_matcher)
        probe_decision = probe.decide(transcript.text, now_s=now_s, kws_match=kws_match)
        if probe_decision.decision != "NO_ACTION":
            candidates.append((transcript.text, kws_match, probe_decision))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2].decision == "TRIGGER_ALL", c[2].confidence))
    return candidates[-1]


def separated_extra_event(pcm, decision, extras, stt, engine, separator, now_s, kws=None):
    """WeSep when per-sentence matching (UDKEngine.decide_all) already fired
    extra events in this segment -- previously WeSep was skipped there, which
    blocked it on the key prepped_data overlap clips (2026-09-23). No
    restore() here (it would drop the sentence events); instead the
    separated stream's candidate is ADDED as one more event when:
      - the whole-segment decision was NO_ACTION (same bar as
        retry_with_separation), or it was a non-exact match and the
        candidate is EXACT (same bar as recheck_verify_with_separation), and
      - its UDK hasn't already fired in this segment, so one spoken phrase
        heard twice (mixture + separated voice) never counts as a repeat.
    Returns the new event or None."""
    if decision.layer == "exact" and decision.decision != "NO_ACTION":
        return None
    best = _best_candidate(pcm, stt, engine, separator, now_s, kws)
    if best is None or best[2].udk is None:
        return None
    fired = {e.udk.udk_id for e in [decision, *extras] if e.decision != "NO_ACTION" and e.udk}
    if best[2].udk.udk_id in fired:
        return None
    if decision.decision != "NO_ACTION" and best[2].layer != "exact":
        return None
    return engine.decide(best[0], now_s=now_s, kws_match=best[1])


def recheck_verify_with_separation(pcm, decision, pre_decision_state, stt, engine, separator, now_s, kws=None):
    """Second WeSep use (2026-09-23): the overlap eval found 9/10 remaining
    misses were a TRIGGER_VERIFY on the WRONG UDK -- the other person's
    ordinary sentence matched weakly (fuzzy/semantic/kws) before the
    NO_ACTION-only fallback could run. Rule: only a non-exact VERIFY is
    re-checked, and it is replaced only by an EXACT match on the separated
    stream for a DIFFERENT UDK -- the user verbatim saying a UDK outranks a
    weak match. Otherwise the original decision stands (never vetoed).
    pre_decision_state = engine.snapshot() taken before `decision` was
    made, restored before re-deciding so one segment counts once."""
    if decision.decision != "TRIGGER_VERIFY" or decision.layer == "exact":
        return decision
    best = _best_candidate(pcm, stt, engine, separator, now_s, kws)
    if best is None or best[2].layer != "exact" or best[2].udk is None or best[2].udk == decision.udk:
        return decision
    engine.restore(pre_decision_state)
    return engine.decide(best[0], now_s=now_s, kws_match=best[1])
