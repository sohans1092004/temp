"""
M6: at-rest encryption + key management (Section 10's baseline controls).

A local versioned-key file stands in for a managed KMS here -- same
reasoning as storage.py/db.py/event_delivery.py's other local stand-ins:
a real KMS needs cloud infra this prototype has no reason to stand up,
and MultiFernet's key-rotation model (encrypt with the newest key, decrypt
by trying all known keys) is exactly the semantics a real KMS's key
versions give you, so swapping this for a real KMS client later is a
backend change, not a redesign.

Uses `cryptography` (audited, industry-standard) rather than hand-rolled
AES -- this is the one place in the project where reaching for a library
instead of writing it yourself is the correct call, not a laziness
violation (ponytail's own rule: never simplify away security measures).

Section 10 also asks for separate key material for audio-at-rest vs.
general app secrets -- callers get that by constructing two KeyManagers
against two different key files, not by this module tracking "purpose."
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.fernet import Fernet, MultiFernet


class KeyManager:
    def __init__(self, keys_path: str | Path):
        self.keys_path = Path(keys_path)
        self._keys = self._load_or_create()
        self._fernet = MultiFernet([Fernet(k.encode()) for k in self._keys])

    def _load_or_create(self) -> list[str]:
        if self.keys_path.exists():
            return json.loads(self.keys_path.read_text())
        self.keys_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [Fernet.generate_key().decode()]
        self.keys_path.write_text(json.dumps(keys))
        return keys

    def rotate(self) -> None:
        """New key becomes current (used for all new encryption); old
        keys are kept so data encrypted under them still decrypts
        (Section 10: "key rotation on a schedule")."""
        self._keys.insert(0, Fernet.generate_key().decode())
        self.keys_path.write_text(json.dumps(self._keys))
        self._fernet = MultiFernet([Fernet(k.encode()) for k in self._keys])

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._fernet.decrypt(ciphertext)

    def encrypt_str(self, plaintext: str) -> str:
        return self.encrypt(plaintext.encode()).decode()

    def decrypt_str(self, ciphertext: str) -> str:
        return self.decrypt(ciphertext.encode()).decode()
