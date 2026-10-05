"""
lattice.py
----------
A from-scratch, reference (non-optimised) implementation of a Fiat-Shamir-with-Aborts
lattice signature core, as used by the LGAN-I / LGAN-II schemes in the manuscript
"Adversarial Machine Learning Threat Analysis for GAN-Augmented Post-Quantum
Lattice-Based Signature Schemes" (Faridi, Chaudhary, Masood).

This models Gates 1 and 2 of the LGAN construction:
    Gate 1: ||z||_inf < gamma1 - beta            (bound / rejection-sampling check)
    Gate 2: H(w' , M) == c                        (Fiat-Shamir consistency check)

Polynomial multiplication is approximated with numpy.convolve (mod q, truncated to
degree n), exactly as described in the manuscript's implementation notes for the
Dilithium2 reference. This is NOT a constant-time, side-channel-safe, or otherwise
production-grade implementation -- it exists purely to support the adversarial-ML
evaluation described in the paper.
"""

from __future__ import annotations
import hashlib
import numpy as np
from dataclasses import dataclass


def _poly_mul_mod(a: np.ndarray, b: np.ndarray, q: int, n: int) -> np.ndarray:
    """Negacyclic-ish polynomial multiplication approximation via convolve + reduction mod x^n+1.

    True Dilithium reduction uses x^n + 1 (negacyclic convolution). We approximate this
    with linear convolution followed by folding the upper half back with a sign flip,
    which reproduces the negacyclic reduction exactly for the schoolbook case.
    """
    conv = np.convolve(a, b).astype(np.int64)
    # conv has length 2n-1; fold indices >= n back with a minus sign (x^n = -1)
    result = conv[:n].copy()
    if len(conv) > n:
        upper = conv[n:]
        result[: len(upper)] -= upper
    return np.mod(result, q)


def _hash_challenge(w: np.ndarray, message: bytes, n: int, kappa: int, seed_extra: bytes = b"") -> np.ndarray:
    """Fiat-Shamir challenge c: a sparse {-1,0,1}^n vector with exactly `kappa` nonzero entries.

    Derived deterministically from H(w || M) so verifier and signer agree.
    """
    digest_seed = hashlib.sha3_256(w.astype(np.int64).tobytes() + message + seed_extra).digest()
    rng = np.random.default_rng(int.from_bytes(digest_seed[:8], "little"))
    c = np.zeros(n, dtype=np.int64)
    positions = rng.choice(n, size=kappa, replace=False)
    signs = rng.choice([-1, 1], size=kappa)
    c[positions] = signs
    return c


@dataclass
class LatticeParams:
    name: str
    n: int              # ring / vector dimension
    q: int               # modulus
    eta: int             # secret key coefficient bound (uniform in [-eta, eta])
    gamma1: int          # masking vector bound
    beta: int            # kappa * eta (max shift introduced by c*s)
    kappa: int           # challenge weight (# of nonzero entries in c)


def make_params(level: str) -> LatticeParams:
    """Parameter sets matching Table I of the manuscript (LGAN-I ~ Dilithium2-level,
    LGAN-II ~ higher security level)."""
    if level == "LGAN-I":
        return LatticeParams(name="LGAN-I", n=256, q=8_380_417, eta=2,
                              gamma1=1 << 17, beta=39 * 2, kappa=39)
    elif level == "LGAN-II":
        return LatticeParams(name="LGAN-II", n=384, q=8_380_417, eta=2,
                              gamma1=1 << 19, beta=49 * 2, kappa=49)
    elif level == "Dilithium2":
        return LatticeParams(name="Dilithium2", n=256, q=8_380_417, eta=2,
                              gamma1=1 << 17, beta=39 * 2, kappa=39)
    else:
        raise ValueError(f"unknown parameter level {level}")


class LatticeCore:
    """Fiat-Shamir with Aborts signature core (Gates 1 & 2 only -- no GAN gate)."""

    def __init__(self, params: LatticeParams, rng: np.random.Generator | None = None):
        self.p = params
        self.rng = rng or np.random.default_rng()
        self.A = self.rng.integers(0, self.p.q, size=(self.p.n,), dtype=np.int64)
        self.s: np.ndarray | None = None
        self.t: np.ndarray | None = None

    # ---------------------------------------------------------------- keygen
    def keygen(self):
        p = self.p
        self.s = self.rng.integers(-p.eta, p.eta + 1, size=(p.n,)).astype(np.int64)
        self.t = _poly_mul_mod(self.A, self.s, p.q, p.n)
        return self.A, self.t, self.s

    # ------------------------------------------------------------------ sign
    def sign_core(self, message: bytes, max_attempts: int = 200, seed_extra: bytes = b""):
        """Runs the abort loop and returns (z, c, w) satisfying Gates 1 & 2, or None."""
        p = self.p
        for _ in range(max_attempts):
            y = self.rng.integers(-p.gamma1, p.gamma1 + 1, size=(p.n,)).astype(np.int64)
            w = _poly_mul_mod(self.A, y, p.q, p.n)
            c = _hash_challenge(w, message, p.n, p.kappa, seed_extra)
            cs = _poly_mul_mod(c, self.s, p.q, p.n)
            z = y + cs
            # centre z into (-q/2, q/2] before bound-checking (rejection sampling)
            z_centered = np.mod(z + p.q // 2, p.q) - p.q // 2
            if np.max(np.abs(z_centered)) < p.gamma1 - p.beta:
                return z_centered, c, w
        return None  # signing failed after max_attempts aborts (astronomically unlikely)

    # ---------------------------------------------------------------- verify
    def verify_core(self, message: bytes, z: np.ndarray, c: np.ndarray, seed_extra: bytes = b"") -> tuple[bool, bool]:
        """Returns (gate1_ok, gate2_ok)."""
        p = self.p
        gate1_ok = bool(np.max(np.abs(z)) < p.gamma1 - p.beta)
        ct = _poly_mul_mod(c, self.t, p.q, p.n)
        w_prime = np.mod(_poly_mul_mod(self.A, z, p.q, p.n) - ct, p.q)
        c_prime = _hash_challenge(w_prime, message, p.n, p.kappa, seed_extra)
        gate2_ok = bool(np.array_equal(c, c_prime))
        return gate1_ok, gate2_ok
