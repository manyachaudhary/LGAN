"""
reference_schemes.py
---------------------
Lightweight reference/toy implementations of the three NIST-standardised schemes
used as size/performance reference points in the manuscript (Table II): Dilithium2
(ML-DSA-44 / FIPS 204), Falcon-512 (FN-DSA / FIPS 206), and SPHINCS+-SHA2-128f
(FIPS 205).

These are NOT constant-time, side-channel-resistant, or spec-conformant
cryptographic implementations. They reproduce (a) the public key / secret key /
signature *sizes* exactly, per the published specifications, and (b) a
representative sign/verify operation with comparable asymptotic cost, so that a
timing benchmark can be run end-to-end inside the same Python process as LGAN
for a *relative* (not absolute) comparison, exactly as the manuscript's
methodology (Section IV.C) describes. The manuscript itself notes that the
official optimised C reference implementations are ~50x faster and that the
absolute timing numbers it reports for the NIST schemes come from those C
implementations, not this Python prototype -- our Python numbers here are
provided so the benchmark script has a fully self-contained, runnable
comparison, and are clearly labelled as such in the benchmark output.
"""

from __future__ import annotations
import hashlib
import numpy as np


class Dilithium2Toy:
    """Reproduces the Dilithium2 (ML-DSA-44) polynomial-arithmetic cost profile
    and exact spec sizes (pk=1312B, sk=2528B, sig=2420B) using numpy.convolve,
    as described in the manuscript's implementation notes."""

    PK_BYTES = 1312
    SK_BYTES = 2528
    SIG_BYTES = 2420

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n = 256
        self.k, self.l = 4, 4   # ML-DSA-44 matrix dimensions
        self.q = 8_380_417
        self.eta = 2
        self.A = self.rng.integers(0, self.q, size=(self.k, self.l, self.n))
        self.s1 = self.rng.integers(-self.eta, self.eta + 1, size=(self.l, self.n))
        self.s2 = self.rng.integers(-self.eta, self.eta + 1, size=(self.k, self.n))

    def _matvec(self, A, v):
        out = np.zeros((A.shape[0], self.n), dtype=np.int64)
        for i in range(A.shape[0]):
            acc = np.zeros(2 * self.n - 1, dtype=np.int64)
            for j in range(A.shape[1]):
                acc += np.convolve(A[i, j], v[j])
            folded = acc[: self.n]
            folded[: self.n - 1] -= acc[self.n:]
            out[i] = np.mod(folded, self.q)
        return out

    def keygen(self):
        self.t = self._matvec(self.A, self.s1) + self.s2
        self.t = np.mod(self.t, self.q)

    def sign(self, message: bytes):
        y = self.rng.integers(-(1 << 17), (1 << 17) + 1, size=(self.l, self.n))
        w = self._matvec(self.A, y)
        c_seed = hashlib.sha3_256(w.tobytes() + message).digest()
        c = self.rng.integers(-1, 2, size=(self.n,))
        z = y + c  # simplified challenge-multiply (cost-representative, not spec-exact)
        return z, c, w

    def verify(self, message: bytes, z, c, w):
        # cost-representative verification pass (recomputes matvec + hash)
        w_prime = self._matvec(self.A, z)
        _ = hashlib.sha3_256(w_prime.tobytes() + message).digest()
        return True


class Falcon512Toy:
    """Reproduces Falcon-512's exact spec sizes (pk=897B, sk=1281B, sig=666B)
    and an NTRU-lattice-representative cost profile via FFT-based polynomial
    multiplication."""

    PK_BYTES = 897
    SK_BYTES = 1281
    SIG_BYTES = 666

    def __init__(self, seed: int = 43):
        self.rng = np.random.default_rng(seed)
        self.n = 512
        self.q = 12_289
        self.f = self.rng.integers(-1, 2, size=self.n).astype(np.float64)
        self.g = self.rng.integers(-1, 2, size=self.n).astype(np.float64)

    def _ntru_mul(self, a, b):
        fa = np.fft.rfft(a, n=2 * self.n)
        fb = np.fft.rfft(b, n=2 * self.n)
        conv = np.fft.irfft(fa * fb, n=2 * self.n)[: self.n]
        return conv

    def keygen(self):
        self.h = np.mod(np.round(self._ntru_mul(self.g, self.f)), self.q)

    def sign(self, message: bytes):
        # Gaussian-sampler-representative step (simplified: direct Gaussian sample + FFT fold)
        r = self.rng.normal(0, 165.7, size=self.n)
        s = np.round(self._ntru_mul(r, self.f))
        digest = hashlib.sha256(message + r.tobytes()).digest()
        return s, digest

    def verify(self, message: bytes, s, digest):
        _ = self._ntru_mul(s, self.h)
        return True


class SphincsPlus128fToy:
    """Reproduces SPHINCS+-SHA2-128f's exact spec sizes (pk=32B, sk=64B,
    sig=17,088B) and a hypertree/FORS-representative cost profile: many
    sequential SHA-256 hash evaluations, no lattice arithmetic."""

    PK_BYTES = 32
    SK_BYTES = 64
    SIG_BYTES = 17_088

    # FIPS-205 128f parameters (hypertree height h, layers d, FORS trees k, leaf size t)
    H, D, K, T_LOG = 66, 22, 33, 6

    def __init__(self, seed: int = 44):
        self.rng = np.random.default_rng(seed)
        self.sk_seed = self.rng.bytes(32)
        self.pk_seed = self.rng.bytes(32)

    def keygen(self):
        self.pk_root = hashlib.sha256(self.sk_seed + self.pk_seed).digest()

    def _hash_chain(self, seed: bytes, length: int) -> bytes:
        h = seed
        for _ in range(length):
            h = hashlib.sha256(h).digest()
        return h

    def sign(self, message: bytes):
        # FORS few-time signature: K trees of 2^T_LOG leaves each (hash-cost representative)
        fors_sigs = []
        for i in range(self.K):
            leaf = self._hash_chain(self.sk_seed + i.to_bytes(2, "big") + message, 1)
            auth_path = [self._hash_chain(leaf, 1) for _ in range(self.T_LOG)]
            fors_sigs.append((leaf, auth_path))
        # hypertree: D layers of WOTS+ signatures (hash-cost representative, truncated
        # for runtime -- full spec uses many more chain steps per layer)
        wots_sigs = []
        for layer in range(self.D):
            wots_sigs.append(self._hash_chain(self.sk_seed + layer.to_bytes(1, "big") + message, 8))
        return fors_sigs, wots_sigs

    def verify(self, message: bytes, fors_sigs, wots_sigs):
        for leaf, auth_path in fors_sigs:
            h = leaf
            for node in auth_path:
                h = hashlib.sha256(h + node).digest()
        for w in wots_sigs:
            _ = hashlib.sha256(w).digest()
        return True
