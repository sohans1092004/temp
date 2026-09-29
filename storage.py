"""
Chunked audio durability layer (Section 6): each chunk gets a checksum,
gaps are recorded explicitly rather than silently smoothed, and the full
recording is the ordered concatenation of segments by sequence number.

Local-filesystem-backed for this prototype -- a stand-in for the real
object storage / ingest-node disk queue split from Section 6, which needs
an actual client/server boundary to put a durability boundary between
(that's M3+). The interface (write_segment/finalize/read_full_audio) is
what should stay the same if the backing store is later swapped for real
object storage.

ponytail: the manifest is one JSON file rewritten in full on every
write_segment call (O(n) per frame). Fine at prototype/test-journey
scale (minutes); move to an append-only log or a DB row per segment
(M5) if journeys run long enough for this to show up in profiling.

M6: an optional KeyManager encrypts audio at rest (Section 10). The
checksum in SegmentMeta is always over the plaintext -- Fernet encryption
isn't deterministic, so checksumming ciphertext would make a legitimate
retry of the exact same audio look like "different content" and wrongly
trip the idempotency check above. On-disk corruption is instead caught by
Fernet's own authentication tag (decrypt fails outright on tampering),
checked defense-in-depth against the plaintext checksum after decrypting.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SegmentMeta:
    seq: int
    checksum: str
    byte_length: int
    written_at: float


@dataclass
class JourneyManifest:
    journey_id: str
    segments: list[SegmentMeta]
    missing_seqs: list[int]


class AudioStore:
    def __init__(self, base_dir: str | Path, key_manager=None):
        self.base_dir = Path(base_dir)
        self._key_manager = key_manager

    def _encrypt(self, data: bytes) -> bytes:
        return self._key_manager.encrypt(data) if self._key_manager is not None else data

    def _decrypt(self, data: bytes) -> bytes:
        return self._key_manager.decrypt(data) if self._key_manager is not None else data

    def _journey_dir(self, journey_id: str) -> Path:
        d = self.base_dir / journey_id
        (d / "segments").mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self, journey_id: str) -> Path:
        return self._journey_dir(journey_id) / "manifest.json"

    def _load_manifest(self, journey_id: str) -> list[dict]:
        path = self._manifest_path(journey_id)
        return json.loads(path.read_text()) if path.exists() else []

    def write_segment(self, journey_id: str, seq: int, pcm: bytes) -> SegmentMeta:
        """Durability boundary: the chunk is on disk with a verifiable
        checksum before this returns, so nothing upstream can lose it on a
        subsequent crash (Section 6's ingest-node disk-queue idea, applied
        directly here since there's no separate ingest process yet).

        Idempotent on (journey_id, seq): a client retry after a lost ack
        (Section 3) re-delivers a seq the store already has -- that's a
        no-op, not a second write, or read_full_audio would double-count
        it. A different checksum for the same seq isn't a retry, it's a
        real anomaly, and is raised rather than silently overwritten."""
        checksum = hashlib.sha256(pcm).hexdigest()
        entries = self._load_manifest(journey_id)
        existing = next((e for e in entries if e["seq"] == seq), None)
        if existing is not None:
            if existing["checksum"] != checksum:
                raise ValueError(f"segment {seq} for journey {journey_id} received twice with different content")
            return SegmentMeta(**existing)

        seg_path = self._journey_dir(journey_id) / "segments" / f"{seq:06d}.pcm"
        seg_path.write_bytes(self._encrypt(pcm))
        meta = SegmentMeta(seq=seq, checksum=checksum, byte_length=len(pcm), written_at=time.time())
        entries.append(asdict(meta))
        self._manifest_path(journey_id).write_text(json.dumps(entries, indent=2))
        return meta

    def read_contiguous_prefix(self, journey_id: str) -> tuple[bytes, int]:
        """The audio and next-expected-seq for everything stored
        contiguously from seq=0 -- what a restarted process rebuilds its
        live buffer from after a crash, instead of reprocessing
        everything or skipping segments (Section 6)."""
        manifest = self.finalize(journey_id)
        seg_dir = self._journey_dir(journey_id) / "segments"
        buffer = bytearray()
        next_seq = 0
        for seg in manifest.segments:
            if seg.seq != next_seq:
                break
            buffer.extend(self._decrypt((seg_dir / f"{seg.seq:06d}.pcm").read_bytes()))
            next_seq += 1
        return bytes(buffer), next_seq

    def finalize(self, journey_id: str) -> JourneyManifest:
        """Gaps are recorded, not silently smoothed over (Section 6)."""
        entries = sorted(self._load_manifest(journey_id), key=lambda e: e["seq"])
        segments = [SegmentMeta(**e) for e in entries]
        seqs = [s.seq for s in segments]
        missing = sorted(set(range(seqs[0], seqs[-1] + 1)) - set(seqs)) if seqs else []
        return JourneyManifest(journey_id=journey_id, segments=segments, missing_seqs=missing)

    def read_full_audio(self, journey_id: str) -> bytes:
        """Ordered concatenation by sequence number. Raises on checksum
        mismatch rather than silently returning corrupt audio -- callers
        should check finalize().missing_seqs rather than assume gapless."""
        manifest = self.finalize(journey_id)
        seg_dir = self._journey_dir(journey_id) / "segments"
        chunks = []
        for seg in manifest.segments:
            on_disk = (seg_dir / f"{seg.seq:06d}.pcm").read_bytes()
            try:
                data = self._decrypt(on_disk)
            except Exception as exc:  # decryption failure = tampered/corrupt (Fernet is authenticated)
                raise ValueError(f"segment {seg.seq} checksum mismatch: corrupt on disk") from exc
            if hashlib.sha256(data).hexdigest() != seg.checksum:
                raise ValueError(f"segment {seg.seq} checksum mismatch: corrupt on disk")
            chunks.append(data)
        return b"".join(chunks)

    def delete_journey(self, journey_id: str) -> None:
        """M6's retention sweep (Section 7): removes a journey's audio
        entirely once its retention window has passed. Not undoable by
        design -- retention deletion is meant to be permanent."""
        shutil.rmtree(self.base_dir / journey_id, ignore_errors=True)
