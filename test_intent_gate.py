"""intent_gate wiring, no models: evaluate_full_system.apply_intent_gate (evals) and
api._intent_veto (live) with a stub scorer. python test_intent_gate.py"""
from types import SimpleNamespace as N

import intent_gate
from evaluate_full_system import apply_intent_gate
from udk_engine import DecisionEvent
from udks import UDK

gate = intent_gate.IntentGate.__new__(intent_gate.IntentGate)
gate.threshold = intent_gate.THRESHOLD
gate.p_distress = lambda t, c="": 0.001 if "don't need" in t else 0.999  # stub: negation = not distress
gen = UDK("UDK_05", "I need help right now", "GENERAL")


def ev(t, layer="semantic"):
    return DecisionEvent(decision="TRIGGER_ALL", udk=gen, confidence=0.9, transcript=t, layer=layer, repeated=False, reason="")


# --- evals: both paths, per segment
w1, w2, w3, p1 = ev("I don't need help right now"), ev("Help me please"), ev("You do me!", "fuzzy"), ev("I don't need help right now.")
out = {"whisper": ([w1, w2, w3], [
    {"start_ms": 0, "transcript": "I don't need help right now", "decision": "TRIGGER_ALL", "_event": w1},
    {"start_ms": 5000, "transcript": "Help me please", "decision": "TRIGGER_ALL", "_event": w2},
    {"start_ms": 9000, "transcript": "You do me!", "decision": "TRIGGER_ALL", "_event": w3}], 0),
    "parakeet": ([p1], [
        {"start_ms": 0, "transcript": "I don't need help right now.", "decision": "TRIGGER_ALL", "_event": p1},
        {"start_ms": 5000, "transcript": "Help me, please.", "decision": "NO_ACTION", "_event": None},
        {"start_ms": 9000, "transcript": "You didn't mean.", "decision": "NO_ACTION", "_event": None}], 0)}
apply_intent_gate(out, gate)
wev, wrows, _ = out["whisper"]
assert [e.transcript for e in wev] == ["Help me please", "You do me!"], wev  # negation vetoed; danger kept
assert wrows[0]["decision"] == "NO_ACTION" and wrows[0]["veto_p"] == 0.001
assert out["parakeet"][0] == []  # vetoed on the other path too
assert wrows[2]["decision"] == "TRIGGER_ALL"  # STTs disagree -> never vetoed

# --- live API: whole-segment decision keeps its slot as NO_ACTION, extras are dropped
import api  # noqa: E402

api._intent_gate = gate
res = api._intent_veto([ev("I don't need help right now"), ev("Help me please")], "I don't need help right now, help me please")
assert res[0].decision == "NO_ACTION" and "intent veto" in res[0].reason and res[1].transcript == "Help me please", res
res = api._intent_veto([ev("Help me"), ev("I don't need help right now")], "help me i don't need help right now")
assert [d.transcript for d in res] == ["Help me"], res
assert api._intent_veto([ev("I don't need help right now")], "")[0].decision == "TRIGGER_ALL"  # no second STT text

# --- v2: context + one-word rule, same result on the eval path and the live path
ctx_gate = intent_gate.IntentGate.__new__(intent_gate.IntentGate)
ctx_gate.threshold, ctx_gate.context_s, ctx_gate.short_support = intent_gate.THRESHOLD, 20, 0.9
ctx_gate.p_distress = lambda t, c="": 0.95 if "coming" in c else 0.5


def eval_run(before):
    b = ev("back.", "exact/sentence")
    o = {"whisper": ([b], [
        {"start_ms": 0, "end_ms": 3000, "transcript": before, "decision": "NO_ACTION", "_event": None},
        {"start_ms": 4000, "end_ms": 5000, "transcript": "back.", "decision": "TRIGGER_ALL", "_event": b}], 0),
        "parakeet": ([], [{"start_ms": 4000, "end_ms": 5000, "transcript": "", "decision": "NO_ACTION", "_event": None}], 0)}
    apply_intent_gate(o, ctx_gate)
    return o["whisper"][0]


assert len(eval_run("he is coming closer")) == 1  # context supports the one word: alert
assert eval_run("that was a nice chess move") == []  # it doesn't: silenced

api._intent_gate = ctx_gate
j = N(recent_text=[(1000, "old stuff", ""), (29000, "he is coming closer", "")])
earlier = api._context_segments(j, 30000, True)
assert earlier == ["he is coming closer"] and len(j.recent_text) == 1, (earlier, j.recent_text)  # 20 s window pruned
assert api._intent_veto([ev("back.", "exact/sentence")], "", "back.", earlier)[0].decision == "TRIGGER_ALL"
assert api._intent_veto([ev("back.", "exact/sentence")], "", "back.", ["nice chess move"])[0].decision == "NO_ACTION"
assert api._context_segments(N(recent_text=[(29000, "", "he is coming")]), 30000, True) == ["he is coming"]  # other STT fallback
print("test_intent_gate: all checks passed")
