"""python -m intent_data.test_mine_dialogue  (no downloads)"""
from udks import general_udks

from .mine_dialogue import candidates, heldout, pick

udks = general_udks("en")
film = [("m1", ["You're coming with me.", "Nobody can hear you.", "Let go of me!", "Nice weather today."])]
rows = candidates(film, "cornell", udks)
assert [(r["text"], r["udk_id"]) for r in rows] == [("Let go of me!", "UDK_10")], rows
assert rows[0]["context"] == "You're coming with me. Nobody can hear you."  # attacker's lines, not the reply after
assert rows[0]["category"] == "mined_cornell" and rows[0]["doc"] == "m1"

many = [{"udk_id": "A"}] * 50 + [{"udk_id": "B"}] * 2
got = pick(many, 6)
assert len(got) == 6 and sum(r["udk_id"] == "B" for r in got) == 2  # rare UDK kept, not crowded out

assert heldout("m1") == heldout("m1") and 0 < sum(heldout(f"m{i}") for i in range(1000)) < 400
print("ok")
