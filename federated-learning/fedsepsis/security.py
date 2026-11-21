"""Security layer of the central server.

Every hospital client gets its own secret key when it registers. Before an update leaves the
hospital it is signed with HMAC-SHA256 over its exact bytes; the server recomputes the
signature and drops any update that was changed in transit or sent by an unknown client.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

import numpy as np


def state_bytes(state) -> bytes:
    return b"".join(k.encode() + np.ascontiguousarray(state[k].cpu().numpy()).tobytes()
                    for k in sorted(state))


def digest(state) -> str:
    return hashlib.sha256(state_bytes(state)).hexdigest()


class SecurityLayer:
    def __init__(self):
        self._keys = {}

    def register(self, client_id: str) -> bytes:
        """Issued once to the hospital; kept secret by both sides."""
        self._keys[client_id] = secrets.token_bytes(32)
        return self._keys[client_id]

    @staticmethod
    def sign(key: bytes, state, n_samples: int) -> str:
        msg = state_bytes(state) + str(int(n_samples)).encode()
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    def verify(self, client_id: str, state, n_samples: int, signature: str) -> tuple[bool, str]:
        key = self._keys.get(client_id)
        if key is None:
            return False, "unknown client"
        ok = hmac.compare_digest(self.sign(key, state, n_samples), signature)
        return ok, "" if ok else "signature mismatch (update changed in transit)"
