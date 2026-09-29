"""
M2 checks: personal UDK enrollment (enrollment.py) and audio durability
storage (storage.py), plus the M2 acceptance criterion from Section 14 --
"personal UDK detected reliably across a re-recorded/varied version of the
enrollment phrase, not just the exact enrollment sample."
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from enrollment import enroll
from stt import MockSTT
from storage import AudioStore
from udk_engine import UDKEngine
from udks import GENERAL_UDKS

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        print(f"[FAIL] {name} {detail}")


print("=== Enrollment: distinctiveness/intelligibility validation ===")

good = enroll("U1001", "the weather in Denver is lovely")
check("distinctive phrase, no sample audio -> succeeds", good.ok, str(good.errors))

too_short = enroll("U1001", "help me")
check("too-short phrase -> rejected", not too_short.ok and any("short" in e for e in too_short.errors))

too_generic = enroll("U1001", "I need to go now")
check("phrase matching ordinary conversation -> rejected", not too_generic.ok, str(too_generic.errors))

mock = MockSTT()
sample_pcm = b"\x00\x01" * 100
mock.register(sample_pcm, "the weather in denver is lovely")
with_good_sample = enroll("U1001", "the weather in Denver is lovely", sample_pcm=sample_pcm, stt=mock)
check("sample audio matches phrase -> succeeds", with_good_sample.ok, str(with_good_sample.errors))

mock2 = MockSTT()
bad_sample_pcm = b"\x02\x03" * 100
mock2.register(bad_sample_pcm, "completely unrelated words entirely")
with_bad_sample = enroll("U1001", "the weather in Denver is lovely", sample_pcm=bad_sample_pcm, stt=mock2)
check(
    "sample audio doesn't match phrase -> rejected",
    not with_bad_sample.ok and any("didn't clearly match" in e for e in with_bad_sample.errors),
)

print(f"\n{passed}/{passed + failed} enrollment checks passed so far.\n")

print("=== M2 acceptance: personal UDK detected across varied phrasing ===")

enrolled = enroll("U2002", "purple elephants dance at midnight")
check("enrollment for acceptance test succeeds", enrolled.ok, str(enrolled.errors))
personal_udk = enrolled.udk
engine = UDKEngine(GENERAL_UDKS + [personal_udk])

variants = [
    "purple elephants dance at midnight",  # exact
    "purple elefants dance at midnite",  # ASR-noise misspelling
    "um, purple elephants dance at midnight I think",  # padded/hedged
]
for v in variants:
    decision = engine.decide(v, now_s=time.monotonic() + 100)
    ok = decision.udk is not None and decision.udk.udk_id == personal_udk.udk_id and decision.decision != "NO_ACTION"
    check(f'variant "{v}" -> detected as personal UDK', ok, f"got decision={decision.decision} udk={decision.udk}")

print(f"\n{passed}/{passed + failed} checks passed so far.\n")

print("=== Audio storage: chunking, checksums, gap detection ===")

with tempfile.TemporaryDirectory() as tmp:
    store = AudioStore(tmp)
    journey_id = "J1"
    chunks = [f"chunk-{i}".encode() * 10 for i in range(5)]

    for i, c in enumerate(chunks):
        if i == 3:
            continue  # simulate a dropped segment
        store.write_segment(journey_id, seq=i, pcm=c)

    manifest = store.finalize(journey_id)
    check("manifest has 4 of 5 segments", len(manifest.segments) == 4, str(manifest.segments))
    check("gap detection flags the missing seq", manifest.missing_seqs == [3], str(manifest.missing_seqs))

    full_audio = store.read_full_audio(journey_id)
    expected = b"".join(c for i, c in enumerate(chunks) if i != 3)
    check("read_full_audio concatenates present segments in order", full_audio == expected)

    # corrupt a segment on disk and confirm checksum verification catches it
    seg_path = Path(tmp) / journey_id / "segments" / "000000.pcm"
    seg_path.write_bytes(b"corrupted")
    try:
        store.read_full_audio(journey_id)
        check("checksum mismatch is detected", False, "expected ValueError, none raised")
    except ValueError:
        check("checksum mismatch is detected", True)

print(f"\n{'=' * 40}\n{passed}/{passed + failed} TOTAL PASSED" + (" -- ALL PASSED" if failed == 0 else f", {failed} FAILED"))

if failed:
    raise SystemExit(1)
