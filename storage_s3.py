"""
M2's real object-storage swap, per storage.py's own promise: "swapping
the backing store... for real object storage" should be possible without
changing the interface. Same shape as storage.py's AudioStore
(write_segment/read_contiguous_prefix/finalize/read_full_audio/delete_journey),
backed by any S3-compatible endpoint (MinIO here, real AWS S3 in
production -- boto3 doesn't care which).

Layout: {journey_id}/segments/{seq:06d}.pcm per chunk,
{journey_id}/manifest.json for the segment index -- the same two-part
shape storage.py uses locally, just as S3 keys instead of filesystem paths.
"""

from __future__ import annotations

import hashlib
import json
import time

import boto3
from botocore.exceptions import ClientError

from storage import JourneyManifest, SegmentMeta


class S3AudioStore:
    def __init__(self, bucket: str, endpoint_url: str, access_key: str, secret_key: str, key_manager=None):
        self.bucket = bucket
        self._key_manager = key_manager
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        try:
            self._s3.head_bucket(Bucket=bucket)
        except ClientError:
            self._s3.create_bucket(Bucket=bucket)

    def _encrypt(self, data: bytes) -> bytes:
        return self._key_manager.encrypt(data) if self._key_manager is not None else data

    def _decrypt(self, data: bytes) -> bytes:
        return self._key_manager.decrypt(data) if self._key_manager is not None else data

    def _manifest_key(self, journey_id: str) -> str:
        return f"{journey_id}/manifest.json"

    def _segment_key(self, journey_id: str, seq: int) -> str:
        return f"{journey_id}/segments/{seq:06d}.pcm"

    def _load_manifest(self, journey_id: str) -> list[dict]:
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._manifest_key(journey_id))
            return json.loads(obj["Body"].read())
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return []
            raise

    def write_segment(self, journey_id: str, seq: int, pcm: bytes) -> SegmentMeta:
        """Same idempotency guarantee as storage.py's AudioStore: checksum
        is over plaintext (S3 puts aren't deterministic either once
        encrypted), a retried identical frame is a no-op, a genuine
        conflict raises rather than silently overwriting."""
        checksum = hashlib.sha256(pcm).hexdigest()
        entries = self._load_manifest(journey_id)
        existing = next((e for e in entries if e["seq"] == seq), None)
        if existing is not None:
            if existing["checksum"] != checksum:
                raise ValueError(f"segment {seq} for journey {journey_id} received twice with different content")
            return SegmentMeta(**existing)

        self._s3.put_object(Bucket=self.bucket, Key=self._segment_key(journey_id, seq), Body=self._encrypt(pcm))
        meta = SegmentMeta(seq=seq, checksum=checksum, byte_length=len(pcm), written_at=time.time())
        entries.append(
            {"seq": meta.seq, "checksum": meta.checksum, "byte_length": meta.byte_length, "written_at": meta.written_at}
        )
        self._s3.put_object(Bucket=self.bucket, Key=self._manifest_key(journey_id), Body=json.dumps(entries).encode())
        return meta

    def finalize(self, journey_id: str) -> JourneyManifest:
        entries = sorted(self._load_manifest(journey_id), key=lambda e: e["seq"])
        segments = [SegmentMeta(**e) for e in entries]
        seqs = [s.seq for s in segments]
        missing = sorted(set(range(seqs[0], seqs[-1] + 1)) - set(seqs)) if seqs else []
        return JourneyManifest(journey_id=journey_id, segments=segments, missing_seqs=missing)

    def read_contiguous_prefix(self, journey_id: str) -> tuple[bytes, int]:
        manifest = self.finalize(journey_id)
        buffer = bytearray()
        next_seq = 0
        for seg in manifest.segments:
            if seg.seq != next_seq:
                break
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._segment_key(journey_id, seg.seq))
            buffer.extend(self._decrypt(obj["Body"].read()))
            next_seq += 1
        return bytes(buffer), next_seq

    def read_full_audio(self, journey_id: str) -> bytes:
        manifest = self.finalize(journey_id)
        chunks = []
        for seg in manifest.segments:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._segment_key(journey_id, seg.seq))
            on_disk = obj["Body"].read()
            try:
                data = self._decrypt(on_disk)
            except Exception as exc:
                raise ValueError(f"segment {seg.seq} checksum mismatch: corrupt on disk") from exc
            if hashlib.sha256(data).hexdigest() != seg.checksum:
                raise ValueError(f"segment {seg.seq} checksum mismatch: corrupt on disk")
            chunks.append(data)
        return b"".join(chunks)

    def delete_journey(self, journey_id: str) -> None:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{journey_id}/"):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
