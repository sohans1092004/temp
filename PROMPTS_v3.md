# Intent v3 data prompts (you run these in the LLM; save outputs as .jsonl)

Why: v2 silenced real cries when the context was the ATTACKER talking
("You're not leaving!" -> "Get away!" p=0.000). v2's training data had victims describing
danger, almost never an attacker's line followed by the cry.

## 1. Label mined dialogue (intent_data/mined/*.jsonl)

Paste the prompt, then 100 lines of one file at a time.
- `train_<source>.jsonl` outputs -> append to `intent_data/llm/mined_<source>.jsonl`
- `heldout_<source>.jsonl` outputs -> append to `intent_data/heldout/<source>.jsonl` (NEVER into llm/)

```
Each input line is JSON: "context" = what was said just before (may be several people
talking), "text" = the line to judge. Treat every scene as real audio heard by a
personal-safety app, even if it is from a film.

Label the TEXT line:
  1 = the person saying it is in immediate danger RIGHT NOW or is asking for help
      against a threat (being grabbed, followed, threatened, attacked, trapped).
      The threat may only be visible from the context, e.g. context "You're coming
      with me." then text "Let go of me!" = 1.
  0 = anything else: everyday talk, jokes, play, games, sport, arguments without a
      threat, telling a story about the past, quoting, the ATTACKER or bully speaking,
      medical or practical help ("help me carry this"), "call the police" said calmly
      about a past event.
If you honestly cannot tell from the context, leave that line out.

Output: for every line you keep, the SAME JSON object with one added key "label" (0 or
1). Copy context, text, udk_id, category and doc unchanged. One JSON object per line,
no numbering, no code fences, no commentary.
```

## 2. Generate attacker-context pairs (-> intent_data/llm/p5.jsonl)

Run ~15 times, each in a NEW chat, 40 lines each (long outputs repeat themselves);
change the last line to "set 1".."set 15". Append to p5.jsonl.

```
Write training data for a personal-safety app in Andhra Pradesh, India. A phone
microphone hears speech; the app must tell a real cry for help from the same words
said harmlessly.

Write PAIRS. Both lines of a pair have the SAME "text" (the cry); only "context" differs.
"context" = the exact words the OTHER person said just before, as a microphone would hear
them: 1-3 short spoken lines in quotes-free direct speech. NEVER narration or description
("the kids are playing", "we are rehearsing", "a TV scene is on" are all forbidden).

  label 1: the other person is a stranger, harasser, or someone using force or
           secrecy against the speaker. Signals: coercion ("You're not leaving till
           I say."), isolation ("Nobody comes to this side at night."), secrecy
           ("Don't tell your amma."), grabbing/blocking ("I'm not letting go of your
           hand."), luring ("Just sit in the car, I'll drop you, come.").
  label 0: the other person is family, friends, a coach, a shopkeeper... in a normal
           moment -- and they ALSO use commands, so commands alone never mean danger:
           amma "Sit down and eat first, then phone.", brother "Give me the remote, it's
           my turn!", friend "I'm the tickle monster, I'm coming for you!", coach
           "Again, hold his wrist and twist out.", cousin "Boo! Got you!".

Rules:
- At least half of the label-0 contexts must be COMMANDS ("Sit down", "Come here",
  "Give me that", "Get in, we're late") said by family or friends.
- No giveaway words anywhere: no "pretend", "rehearse", "play", "game", "TV", "movie",
  "scene", "haha", "joke", and no "danger", "attack", "knife", "kill", "help", "police"
  in any context. The difference must come from WHO is talking and WHAT they want.
- Every pair uses a different context situation; do not reuse a context sentence or
  a sentence pattern ("Come closer and ...", "Give me your X and ...") more than twice.
- The label-0 context must NOT be the label-1 context with a reason added
  ("Come closer." vs "Come closer, the baby is sleeping." is forbidden). Write a
  different sentence. At least a third of label-0 contexts are SHORT BARE commands with
  no reason at all, the way family really talks: "Come here.", "Phone. Now.", "Sit."
- Every label-1 context carries a real threat signal -- a stranger, isolation, secrecy,
  holding/blocking, refusing to let the speaker leave. "Keep quiet" alone is not enough.
- The context is spoken ONLY by the other person, never by the one who cries, and it
  never contains the cry's words ("Amma, leave me alone, I'm studying" before "Leave me
  alone!" is forbidden).
- Cries ONLY from this list: "Help!", "Help me!", "No!", "Back!", "Stay back!", "Stop!",
  "Leave me!", "Let go!", "Let go of me!", "Don't touch me!", "Get away from me!", "Stay
  away from me!", "Please don't hurt me!", "Someone help me!", "Somebody help me!",
  "I want to go home!", "I need to get out of here!", "Call the police!".
- Casual Indian English, some Telugu words (amma, anna, akka, ra, em), names like Chinni,
  Babu, Ravi, Lakshmi; places like Benz Circle, RTC bus stand, PG hostel, auto stand.

Output one JSON object per line, no code fences, 40 lines (20 pairs):
{"context": "...", "text": "...", "label": 1, "udk_id": "", "category": "attacker_context"}
set 1
```

## After labelling

    python -m intent_data.build_v2        # merges llm/mined_*.jsonl + llm/p5.jsonl
    (then finetune on Kaggle as for v2)

Accept v3 only if, on intent_data/heldout/ (films it never saw), real cries after an
attacker's line keep P(distress) >= the veto threshold, and prepped danger stays 78/84.
