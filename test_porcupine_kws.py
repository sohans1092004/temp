"""Porcupine (pvporcupine, Picovoice) as a candidate third parallel
detection signal alongside STT/KWS.

VERDICT: DEAD END, before any real-audio test -- two independent,
verifiable blockers, not a measured miss:

1. Hard external dependency: `pvporcupine.create()` requires a Picovoice
   `access_key` (a cloud account credential). There is no offline/local
   mode -- confirmed directly from the installed SDK's signature (see
   `main()` below). No key is available in this environment, and
   creating a third-party cloud account is an external, account-level
   decision, not something to do unilaterally on the user's behalf.

2. Architectural mismatch, independent of (1): Porcupine is a
   short-wake-word engine. Its built-in keyword set --
   {'porcupine', 'jarvis', 'computer', 'hey siri', 'ok google',
   'alexa', 'terminator', 'americano', 'blueberry', 'bumblebee',
   'grapefruit', 'grasshopper', 'picovoice', 'hey barista',
   'pico clock', 'hey google'} -- is exclusively short (1-3 dense
   syllables), phonetically-distinct terms chosen specifically because
   they almost never occur in ordinary speech. Custom keywords are
   consumed as pre-trained `.ppn` binary files (built via the Picovoice
   Console's cloud training, not this SDK) and Picovoice's own guidance
   targets the same short-wake-word shape.

   This is the opposite design goal from this project's UDK phrases
   (udks.py), which are explicitly required to be natural, ordinary-
   sounding SENTENCES under real distress ("Call the police, I need
   help", "Someone is trying to hurt me", 6-8 words) -- precisely the
   kind of phrase a wake-word engine is built to reject as a false
   trigger, not spot reliably. Even with an AccessKey, custom-training
   Porcupine on these phrases would be working directly against its
   design assumptions.

Not tested on real_recordings/ as a result -- there is nothing to run:
no key to call `create()` with, and no reason to expect the technique
would generalize to sentence-length phrases even if there were. Left
as a standalone documented dead end, no code wired in.
"""

from __future__ import annotations


def main() -> None:
    import pvporcupine

    sig = str(__import__("inspect").signature(pvporcupine.create))
    print(f"pvporcupine.create signature: {sig}")
    assert "access_key" in sig, "expected access_key to be a required parameter"
    print("Confirmed: access_key is required -- no offline/local mode.")

    print(f"\nBuilt-in keywords (n={len(pvporcupine.KEYWORDS)}): {sorted(pvporcupine.KEYWORDS)}")
    max_words = max(len(k.split()) for k in pvporcupine.KEYWORDS)
    assert max_words <= 2, "expected all built-in keywords to be short (<=2 words)"
    print(f"Confirmed: all built-in keywords are <= {max_words} words -- short wake-word shape.")

    from udks import GENERAL_UDKS

    udk_word_counts = [len(u.phrase.split()) for u in GENERAL_UDKS]
    print(f"\nUDK phrase word counts: {udk_word_counts}")
    print(f"UDK phrase mean word count: {sum(udk_word_counts) / len(udk_word_counts):.1f}")
    assert min(udk_word_counts) > max_words, "expected every UDK phrase to be longer than any built-in keyword"
    print("Confirmed: every UDK phrase is longer than any built-in Porcupine keyword -- architectural mismatch.")


if __name__ == "__main__":
    main()
