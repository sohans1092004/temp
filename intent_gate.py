"""Intent veto for text alerts: a fine-tuned mDeBERTa (intent_data/finetune.py) reads the
alert's transcript and silences it when it is confidently NOT a distress request, e.g.
"I don't need any help", a quote, "don't touch my stuff" -- but ONLY when both STT
engines heard the same words (veto-agree, 2026-09-26). When Whisper and Parakeet
disagree the audio was unclear, the text may not be what was said, and the alert stays.


Opt-in: UDK_INTENT_GATE=<model dir> (a finetune.py output folder); threshold
UDK_INTENT_THRESHOLD, default 0.02 = calibrate_veto.py's pick: the highest threshold that
keeps >= 99.5% of the intent dataset's DEV distress sentences (99.7% on test). A veto must
only fire when the model is confident the words are NOT distress; 0.99 (finetune.py's
detection threshold at 95% recall) silenced real cries on prepped ("don't touch me" p=0.24).
Kaggle, 2026-09-26, at 0.02: prepped danger vetoes 5 -> 1 (an STT mishearing), harmless
alerts 59 -> 43 of 162 vs the fragment rule alone; real_recordings recall same as gate off.
Only GENERAL UDKs' text alerts (exact/fuzzy/semantic) are judged: custom code words are
not distress sentences, and KWS / scream alerts carry no text to judge.

v2 switches (2026-09-27, all OFF by default = the rule above; for a context-trained model):
  UDK_INTENT_CONTEXT_S=<s>      the model also reads what was said in the previous <s> seconds
                                (premise() below -- the SAME input format training uses)
  UDK_INTENT_SINGLE_STT=1       also judge fuzzy/semantic alerts only ONE STT heard (most real
                                false alarms); exact hits still need both STTs to be judged
  UDK_INTENT_SHORT_SUPPORT=<p>  a one-word hit ("back.", "go.", "Help!") alerts only when the
                                model, reading the context, gives P(distress) >= p (user rule:
                                one word is not enough on its own)
"""
from __future__ import annotations

import os

import re

from rapidfuzz import fuzz

HYPOTHESIS = "The speaker is in immediate danger or is asking for help."  # as trained (intent_data/nli_eval.py)
MAX_LEN = 160  # as trained (intent_data/finetune.py)
AGREE = 80  # token_set_ratio: the other STT heard about the same words
THRESHOLD = 0.02  # see module docstring / calibrate_veto.py
TEXT_LAYERS = ("exact", "fuzzy", "semantic")
CONTEXT_WORDS = 60  # context is cut from the FRONT so the current sentence always fits MAX_LEN
SHORT_MAX_WORDS = 1


def premise(text: str, context: str = "") -> str:
    """The model's input. Sentence alone when there is no context (what v1 was trained on)."""
    ctx = " ".join(context.split()[-CONTEXT_WORDS:])
    return f"Before: {ctx} Now: {text}" if ctx else text


def context_before(segment_text: str, alert_text: str, earlier: list[str]) -> str:
    """Everything said before this alert: earlier segments, plus -- for a per-sentence alert --
    the part of its own segment before that sentence."""
    i = segment_text.find(alert_text) if alert_text != segment_text else -1
    return " ".join([*earlier, segment_text[:i] if i > 0 else ""]).strip()


def n_words(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", text.lower()))


def heard_by_both(alert_text: str, other_text: str) -> bool:
    return bool(other_text) and fuzz.token_set_ratio(alert_text.lower(), other_text.lower()) >= AGREE


class IntentGate:
    context_s: float = 0.0
    single_stt: bool = False
    short_support: float | None = None

    def __init__(self, model_dir: str, device: str = "cpu", threshold: float = THRESHOLD):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.threshold = torch, device, threshold
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def p_distress(self, text: str, context: str = "") -> float:
        enc = self.tok([premise(text, context)], [HYPOTHESIS], truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return float(self.torch.softmax(self.model(**enc).logits.float(), -1)[0, 1])

    def veto(self, decision, other_text: str, context: str = "") -> float | None:
        """P(distress) if this alert should be silenced, else None."""
        if (decision.decision == "NO_ACTION" or decision.udk is None or decision.udk.udk_type != "GENERAL"
                or not decision.layer.split("/")[0].startswith(TEXT_LAYERS)):
            return None
        ctx = context if self.context_s else ""
        if self.short_support is not None and n_words(decision.transcript) <= SHORT_MAX_WORDS:
            p = self.p_distress(decision.transcript, ctx)
            return p if p < self.short_support else None
        # single-STT judging covers LOOSE matches only (fuzzy/semantic: where invented English and
        # near-misses come from). An exact hit means the UDK's own words were heard, so it keeps the
        # both-STTs rule: v2 vetoed a real "Get away!" after the attacker's "You're not leaving!"
        loose = not decision.layer.startswith("exact")
        if not heard_by_both(decision.transcript, other_text) and not (self.single_stt and loose):
            return None
        p = self.p_distress(decision.transcript, ctx)
        return p if p < self.threshold else None


def from_env(device: str = "cpu") -> IntentGate | None:
    path = os.environ.get("UDK_INTENT_GATE")
    if not path:
        return None
    if not os.path.isfile(os.path.join(path, "config.json")):
        # a missing model must not take detection down with it: run without the veto, loudly
        import warnings

        warnings.warn(f"UDK_INTENT_GATE={path} has no model (image built without UDK_INTENT_MODEL_REPO?) -- intent gate OFF")
        return None
    gate = IntentGate(path, device=device, threshold=float(os.environ.get("UDK_INTENT_THRESHOLD", THRESHOLD)))
    gate.context_s = float(os.environ.get("UDK_INTENT_CONTEXT_S", 0))
    gate.single_stt = os.environ.get("UDK_INTENT_SINGLE_STT") == "1"
    short = os.environ.get("UDK_INTENT_SHORT_SUPPORT")
    gate.short_support = float(short) if short else None
    print(f"intent gate ON: {path} (threshold {gate.threshold}, {device}; context {gate.context_s:g}s, "
          f"single-STT {'on' if gate.single_stt else 'off'}, one-word support {gate.short_support})", flush=True)
    return gate


if __name__ == "__main__":  # self-check of the no-model parts
    from types import SimpleNamespace as N

    assert heard_by_both("It's fine, I'm safe.", "it's fine, i'm safe, nothing wrong")
    assert not heard_by_both("You do me!", "You didn't mean.")
    assert not heard_by_both("Help me", "")
    g = IntentGate.__new__(IntentGate)
    g.threshold, g.p_distress = THRESHOLD, lambda t, c="": 0.001
    gen, custom = N(udk_type="GENERAL"), N(udk_type="PERSONAL")
    ev = lambda layer, udk=gen, t="I don't need any help": N(decision="TRIGGER_ALL", udk=udk, layer=layer, transcript=t)
    assert g.veto(ev("semantic"), "I don't need any help.") == 0.001
    assert g.veto(ev("exact/sentence"), "I don't need any help.") == 0.001
    assert g.veto(ev("semantic"), "I don't need any hell") == 0.001  # ASR noise still "the same words"
    assert g.veto(ev("semantic"), "") is None  # other STT heard nothing: keep
    assert g.veto(ev("kws"), "I don't need any help") is None  # no text to judge
    assert g.veto(ev("exact", udk=custom), "I don't need any help") is None  # code words exempt
    g.p_distress = lambda t, c="": 0.05  # unsure, not confidently harmless
    assert g.veto(ev("semantic"), "I don't need any help") is None  # model not confident it is harmless: keep
    # v2 switches
    assert premise("Back.") == "Back." and premise("Back.", "he is coming closer") == "Before: he is coming closer Now: Back."
    assert premise("x", " ".join(["w"] * 100)).split().count("w") == CONTEXT_WORDS
    assert context_before("Okay. Let go of me!", "Let go of me!", ["he grabbed me"]) == "he grabbed me Okay."
    assert context_before("Let go of me", "Let go of me", ["a", "b"]) == "a b"
    g.p_distress = lambda t, c="": 0.001
    assert g.veto(ev("semantic"), "something else") is None  # single-STT off by default
    g.single_stt = True
    assert g.veto(ev("semantic"), "something else") == 0.001
    assert g.veto(ev("fuzzy/sentence"), "something else") == 0.001
    assert g.veto(ev("exact/sentence", t="Get away!"), "Nobody's going anywhere.") is None  # exact: both-STT rule
    assert g.veto(ev("exact/sentence", t="Get away!"), "get away") == 0.001  # both heard it: judged as before
    g.single_stt, g.short_support = False, 0.9
    seen = []
    g.p_distress = lambda t, c="": seen.append(c) or (0.95 if "coming" in c else 0.5)
    assert g.veto(ev("exact/sentence", t="back."), "", "chess move") == 0.5  # one word, weak context: silenced
    g.context_s = 20
    assert g.veto(ev("exact/sentence", t="back."), "", "he is coming closer") is None  # strong context: alert
    g.context_s = 0
    assert g.veto(ev("exact/sentence", t="back."), "", "he is coming closer") == 0.5 and seen[-1] == ""  # context off
    assert g.veto(ev("semantic", t="Get away!"), "") is None  # two words: normal rule
    print("intent_gate self-check passed")
