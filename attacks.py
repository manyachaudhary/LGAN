"""
attacks.py
----------
Implements the five-level adversarial attack framework (Section III.B of the
manuscript) against an LGANScheme instance (Design A or Design B).

Each attack function takes a target scheme (already keygen'd) and a target
message, and returns a forged Signature attempt. The caller (security_eval.py)
then runs scheme.verify() on the result to score gate1/gate2/gate3/full.

  L1 - Naive Random Forgery
  L2 - EUF-CMA Oracle Attack (empirical z_gan distribution)
  L3 - Gradient Ascent on Public Discriminator (white-box, Design A only)
  L4 - Surrogate Model / Transfer Attack (black-box query of verify())
  L5 - Combined Joint Attack (oracle warm-start + multi-restart gradient ascent)
"""

from __future__ import annotations
import numpy as np
from gan_gate import Discriminator, DesignA, DesignB
from schemes import Signature, LGANScheme


# ----------------------------------------------------------------------- L1
def attack_l1_random(scheme: LGANScheme, message: bytes, rng: np.random.Generator) -> Signature:
    """Naive random forgery: random z, w, c, z_gan components."""
    p = scheme.lattice.p
    z = rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)
    z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
    if isinstance(scheme.gate3, DesignB):
        pi = rng.bytes(32)
        C_G = scheme.gate3.C_G  # attacker can read pk, but cannot compute a matching pi
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=pi, C_G=C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan)


# ----------------------------------------------------------------------- L2
def attack_l2_euf_cma_oracle(scheme: LGANScheme, message: bytes, rng: np.random.Generator,
                              oracle_queries: int = 50):
    """Query the signing oracle on `oracle_queries` chosen messages, collect genuine
    z_gan samples, fit an empirical Gaussian, and forge on the (unseen) target
    message by sampling from that empirical distribution. z, c, w are still
    forged randomly -- this attack targets Gate 3 in isolation."""
    p = scheme.lattice.p
    collected = []
    for i in range(oracle_queries):
        sig = scheme.sign(f"__l2_oracle_query_{i}__".encode())
        if sig is not None:
            collected.append(sig.z_gan)
    collected = np.stack(collected)
    mu, sigma = collected.mean(axis=0), collected.std(axis=0) + 1e-6

    z_gan_forged = np.clip(rng.normal(mu, sigma), -1, 1)

    z = rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)

    if isinstance(scheme.gate3, DesignB):
        pi = rng.bytes(32)
        return Signature(z=z, c=c, w=w, z_gan=z_gan_forged, pi=pi, C_G=scheme.gate3.C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan_forged)


# ----------------------------------------------------------------------- L3
def attack_l3_gradient_ascent(scheme: LGANScheme, message: bytes, rng: np.random.Generator,
                               steps: int = 300, lr: float = 0.02):
    """White-box gradient ascent on the public discriminator D (Design A only).
    For Design B there is no differentiable public target, so this attack
    degenerates to random guessing (by construction -- Gate 3 is a hash check)."""
    p = scheme.lattice.p
    z = rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)

    if isinstance(scheme.gate3, DesignA):
        feat_z = scheme.gate3.feat(z)
        z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
        D = scheme.gate3.D
        for _ in range(steps):
            score, grad = D.score_and_grad_wrt_zgan(feat_z, z_gan)
            z_gan = np.clip(z_gan + lr * grad, -1, 1)
        return Signature(z=z, c=c, w=w, z_gan=z_gan)
    else:
        # Design B: nothing differentiable to climb -- attacker falls back to random guess
        z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
        pi = rng.bytes(32)
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=pi, C_G=scheme.gate3.C_G)


# ----------------------------------------------------------------------- L4
def _train_shadow_discriminator(scheme: LGANScheme, rng: np.random.Generator,
                                 n_queries: int = 1500):
    """Trains a shadow discriminator D' by querying scheme.verify() (Gate 3 only)
    as a black-box oracle: genuine signatures -> positive examples, random
    z_gan vectors -> negative examples."""
    cfg = scheme.gate3.cfg
    Dp = Discriminator(cfg.feat_dim, cfg.latent_dim, cfg.hidden_dim, rng)

    feats, zgans, labels = [], [], []
    n_each = n_queries // 2
    for i in range(n_each):
        msg = f"__l4_shadow_query_pos_{i}__".encode()
        sig = scheme.sign(msg)
        if sig is None:
            continue
        res = scheme.verify(msg, sig)
        feats.append(scheme.gate3.feat(sig.z))
        zgans.append(sig.z_gan)
        labels.append(1.0 if res["gate3"] else 0.0)
    for i in range(n_each):
        msg = f"__l4_shadow_query_neg_{i}__".encode()
        sig = scheme.sign(msg)
        if sig is None:
            continue
        random_zgan = rng.uniform(-1, 1, size=cfg.latent_dim)
        # query verify() as an oracle with a tampered z_gan
        tampered = Signature(z=sig.z, c=sig.c, w=sig.w, z_gan=random_zgan,
                              pi=(scheme.gate3.sign_proof(random_zgan, sig.w, msg)
                                  if isinstance(scheme.gate3, DesignB) else None),
                              C_G=getattr(scheme.gate3, "C_G", None))
        res = scheme.verify(msg, tampered)
        feats.append(scheme.gate3.feat(sig.z))
        zgans.append(random_zgan)
        labels.append(1.0 if res["gate3"] else 0.0)

    if len(feats) == 0 or len(set(labels)) < 2:
        return Dp  # degenerate: untrained shadow model

    X = np.concatenate([np.stack(feats), np.stack(zgans)], axis=1)
    y = np.array(labels)
    for _ in range(300):
        Dp.train_step(X, y, lr=0.2)
    return Dp


def attack_l4_surrogate_transfer(scheme: LGANScheme, message: bytes, rng: np.random.Generator,
                                  n_queries: int = 1500, steps: int = 300, lr: float = 0.02):
    """Trains a shadow discriminator D' via black-box oracle queries to verify(),
    performs white-box gradient ascent on D', then transfers the resulting z_gan*
    to attack the real scheme."""
    p = scheme.lattice.p
    z = rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)

    Dp = _train_shadow_discriminator(scheme, rng, n_queries=n_queries)
    feat_z = scheme.gate3.feat(z)
    z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
    for _ in range(steps):
        score, grad = Dp.score_and_grad_wrt_zgan(feat_z, z_gan)
        z_gan = np.clip(z_gan + lr * grad, -1, 1)

    if isinstance(scheme.gate3, DesignB):
        pi = rng.bytes(32)
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=pi, C_G=scheme.gate3.C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan)


# ----------------------------------------------------------------------- L5
def attack_l5_combined(scheme: LGANScheme, message: bytes, rng: np.random.Generator,
                        oracle_queries: int = 50, restarts: int = 3, steps: int = 300, lr: float = 0.02):
    """Combined attack: oracle warm-start (empirical z_gan distribution, as in L2),
    multi-restart gradient ascent (as in L3, `restarts` independent runs, best kept),
    and near-genuine z sampled close to the signing distribution."""
    p = scheme.lattice.p

    # oracle warm start
    collected = []
    for i in range(oracle_queries):
        sig = scheme.sign(f"__l5_oracle_query_{i}__".encode())
        if sig is not None:
            collected.append(sig.z_gan)
    collected = np.stack(collected)
    mu, sigma = collected.mean(axis=0), collected.std(axis=0) + 1e-6

    # near-genuine z: sample from a real signing run then jitter it (simulates
    # "signing-distribution insight" without solving the lattice problem)
    genuine_sig = scheme.sign(f"__l5_genuine_template__".encode())
    z = genuine_sig.z.copy() if genuine_sig is not None else \
        rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)

    best_z_gan, best_score = None, -1.0
    if isinstance(scheme.gate3, DesignA):
        D = scheme.gate3.D
        feat_z = scheme.gate3.feat(z)
        for _ in range(restarts):
            z_gan = np.clip(rng.normal(mu, sigma), -1, 1)  # oracle warm start
            for _ in range(steps):
                score, grad = D.score_and_grad_wrt_zgan(feat_z, z_gan)
                z_gan = np.clip(z_gan + lr * grad, -1, 1)
            final_score = D.score(feat_z, z_gan)
            if final_score > best_score:
                best_score, best_z_gan = final_score, z_gan
        return Signature(z=z, c=c, w=w, z_gan=best_z_gan)
    else:
        z_gan = np.clip(rng.normal(mu, sigma), -1, 1)
        pi = rng.bytes(32)
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=pi, C_G=scheme.gate3.C_G)


ATTACKS = {
    "L1": attack_l1_random,
    "L2": attack_l2_euf_cma_oracle,
    "L3": attack_l3_gradient_ascent,
    "L4": attack_l4_surrogate_transfer,
    "L5": attack_l5_combined,
}


# ---------------------------------------------------------------------------
# Cached-setup variants: an attacker trains its oracle statistics / shadow model
# ONCE per target scheme, then reuses that setup across all forgery attempts.
# This is both realistic (a real adversary does not retrain per attempt) and
# necessary for the benchmark to run in reasonable time for L2/L4/L5, which
# involve oracle queries or shadow-model training.
# ---------------------------------------------------------------------------

class AttackerState:
    """Precomputed, reusable attacker knowledge for a given target scheme."""

    def __init__(self, scheme: LGANScheme, rng: np.random.Generator,
                 oracle_queries: int = 50, shadow_queries: int = 400):
        self.scheme = scheme
        # L2/L5 oracle warm start: empirical z_gan distribution
        collected = []
        for i in range(oracle_queries):
            sig = scheme.sign(f"__oracle_setup_query_{i}__".encode())
            if sig is not None:
                collected.append(sig.z_gan)
        collected = np.stack(collected)
        self.mu = collected.mean(axis=0)
        self.sigma = collected.std(axis=0) + 1e-6
        # L4 shadow discriminator (black-box trained once)
        self.shadow_D = _train_shadow_discriminator(scheme, rng, n_queries=shadow_queries)
        # L5 near-genuine template
        self.genuine_template = scheme.sign("__genuine_template__".encode())


def _forge_lattice_junk(scheme: LGANScheme, rng: np.random.Generator):
    """Random (z, c, w) lattice-side components -- fails Gates 1/2 with overwhelming
    probability, isolating the Gate-3 evaluation (used by every attack level)."""
    p = scheme.lattice.p
    z = rng.integers(-p.gamma1 + p.beta + 1, p.gamma1 - p.beta, size=(p.n,)).astype(np.int64)
    c = np.zeros(p.n, dtype=np.int64)
    positions = rng.choice(p.n, size=p.kappa, replace=False)
    c[positions] = rng.choice([-1, 1], size=p.kappa)
    w = rng.integers(0, p.q, size=(p.n,)).astype(np.int64)
    return z, c, w


def forge_l1_cached(scheme: LGANScheme, message: bytes, rng: np.random.Generator, state: AttackerState):
    z, c, w = _forge_lattice_junk(scheme, rng)
    z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
    if isinstance(scheme.gate3, DesignB):
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=rng.bytes(32), C_G=scheme.gate3.C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan)


def forge_l2_cached(scheme: LGANScheme, message: bytes, rng: np.random.Generator, state: AttackerState):
    z, c, w = _forge_lattice_junk(scheme, rng)
    z_gan = np.clip(rng.normal(state.mu, state.sigma), -1, 1)
    if isinstance(scheme.gate3, DesignB):
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=rng.bytes(32), C_G=scheme.gate3.C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan)


def forge_l3_cached(scheme: LGANScheme, message: bytes, rng: np.random.Generator, state: AttackerState,
                     steps: int = 150, lr: float = 0.03):
    z, c, w = _forge_lattice_junk(scheme, rng)
    if isinstance(scheme.gate3, DesignA):
        feat_z = scheme.gate3.feat(z)
        z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
        D = scheme.gate3.D
        for _ in range(steps):
            score, grad = D.score_and_grad_wrt_zgan(feat_z, z_gan)
            z_gan = np.clip(z_gan + lr * grad, -1, 1)
        return Signature(z=z, c=c, w=w, z_gan=z_gan)
    z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
    return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=rng.bytes(32), C_G=scheme.gate3.C_G)


def forge_l4_cached(scheme: LGANScheme, message: bytes, rng: np.random.Generator, state: AttackerState,
                     steps: int = 150, lr: float = 0.03):
    z, c, w = _forge_lattice_junk(scheme, rng)
    feat_z = scheme.gate3.feat(z)
    z_gan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
    Dp = state.shadow_D
    for _ in range(steps):
        score, grad = Dp.score_and_grad_wrt_zgan(feat_z, z_gan)
        z_gan = np.clip(z_gan + lr * grad, -1, 1)
    if isinstance(scheme.gate3, DesignB):
        return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=rng.bytes(32), C_G=scheme.gate3.C_G)
    return Signature(z=z, c=c, w=w, z_gan=z_gan)


def forge_l5_cached(scheme: LGANScheme, message: bytes, rng: np.random.Generator, state: AttackerState,
                     restarts: int = 2, steps: int = 150, lr: float = 0.03):
    z, c, w = _forge_lattice_junk(scheme, rng)
    if state.genuine_template is not None:
        z = state.genuine_template.z.copy()
    if isinstance(scheme.gate3, DesignA):
        D = scheme.gate3.D
        feat_z = scheme.gate3.feat(z)
        best_z_gan, best_score = None, -1.0
        for _ in range(restarts):
            z_gan = np.clip(rng.normal(state.mu, state.sigma), -1, 1)
            for _ in range(steps):
                score, grad = D.score_and_grad_wrt_zgan(feat_z, z_gan)
                z_gan = np.clip(z_gan + lr * grad, -1, 1)
            final_score = D.score(feat_z, z_gan)
            if final_score > best_score:
                best_score, best_z_gan = final_score, z_gan
        return Signature(z=z, c=c, w=w, z_gan=best_z_gan)
    z_gan = np.clip(rng.normal(state.mu, state.sigma), -1, 1)
    return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=rng.bytes(32), C_G=scheme.gate3.C_G)


CACHED_ATTACKS = {
    "L1": forge_l1_cached,
    "L2": forge_l2_cached,
    "L3": forge_l3_cached,
    "L4": forge_l4_cached,
    "L5": forge_l5_cached,
}
