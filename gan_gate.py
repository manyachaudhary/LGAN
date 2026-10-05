"""
gan_gate.py
-----------
Implements the two competing Gate-3 designs from the manuscript:

  Design A ("D-in-pk"): a conditional-GAN discriminator D is trained at keygen
      and published in the public key. Gate 3 verification computes
      D(feat(z), z_gan) > threshold. This is the design shown to be broken by
      gradient ascent (L3-L5).

  Design B ("hash commitment", LGAN-II v2): only a commitment
      C_G = SHA3-256(G.weights) is published. The signer proves knowledge of a
      valid GAN sample via pi = SHA3-256(z_gan || C_G || w || M). Gate 3 reduces
      to a SHA3-256 preimage / second-preimage check (Theorem 1 of the paper).

Generator/Discriminator are small numpy MLPs (this is a reference implementation,
not a production GAN -- the paper's contribution is the *gate design*, not GAN
architecture novelty).
"""

from __future__ import annotations
import hashlib
import numpy as np
from dataclasses import dataclass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _feat(z: np.ndarray, feat_dim: int) -> np.ndarray:
    """Deterministic, differentiable-free feature extraction from the lattice
    response z, projecting it down into a fixed-size real vector in a stable range.
    z is first standardised (scale-invariant) so the projection does not saturate
    regardless of the lattice modulus / masking-bound magnitude."""
    n = z.shape[0]
    z = z.astype(np.float64)
    scale = np.std(z) + 1e-9
    z_norm = z / scale
    proj_rng = np.random.default_rng(1234 + n)
    P = proj_rng.normal(0, 1.0 / np.sqrt(n), size=(feat_dim, n))
    v = P @ z_norm
    return np.tanh(v)


class Generator:
    """Small conditional generator: noise + condition -> z_gan in [-1, 1]^latent_dim."""

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, rng: np.random.Generator):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        in_dim = latent_dim + cond_dim
        self.W1 = rng.normal(0, 0.5, size=(hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.5, size=(latent_dim, hidden_dim))
        self.b2 = np.zeros(latent_dim)

    def forward(self, noise: np.ndarray, cond: np.ndarray) -> np.ndarray:
        x = np.concatenate([noise, cond])
        h = np.tanh(self.W1 @ x + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)
        return out

    def weight_bytes(self) -> bytes:
        return (self.W1.tobytes() + self.b1.tobytes() +
                self.W2.tobytes() + self.b2.tobytes())

    def commitment(self) -> bytes:
        return hashlib.sha3_256(self.weight_bytes()).digest()


class Discriminator:
    """Small MLP discriminator: (feat(z), z_gan) -> score in (0, 1). Fully differentiable
    (analytic gradients are implemented in attacks.py for the L3/L4/L5 gradient-ascent
    attacks)."""

    def __init__(self, feat_dim: int, latent_dim: int, hidden_dim: int, rng: np.random.Generator):
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        in_dim = feat_dim + latent_dim
        W1_feat = rng.normal(0, 0.05, size=(hidden_dim, feat_dim))
        W1_zgan = rng.normal(0, 0.6, size=(hidden_dim, latent_dim))
        self.W1 = np.concatenate([W1_feat, W1_zgan], axis=1)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.5, size=(1, hidden_dim))
        self.b2 = np.zeros(1)

    def score(self, feat_z: np.ndarray, z_gan: np.ndarray) -> float:
        x = np.concatenate([feat_z, z_gan])
        h = np.tanh(self.W1 @ x + self.b1)
        out = sigmoid(self.W2 @ h + self.b2)
        return float(out[0])

    def score_and_grad_wrt_zgan(self, feat_z: np.ndarray, z_gan: np.ndarray):
        """Analytic forward pass + backprop of the score w.r.t. z_gan only.
        Used by the L3 (white-box) gradient-ascent attack."""
        x = np.concatenate([feat_z, z_gan])
        pre1 = self.W1 @ x + self.b1
        h = np.tanh(pre1)
        pre2 = self.W2 @ h + self.b2
        out = sigmoid(pre2)
        score = float(out[0])

        # backprop: d(out)/d(pre2) = out*(1-out)
        d_pre2 = out * (1 - out)  # shape (1,)
        d_h = (self.W2.T @ d_pre2)  # shape (hidden,)
        d_pre1 = d_h * (1 - h ** 2)  # tanh'
        d_x = self.W1.T @ d_pre1     # shape (feat_dim + latent_dim,)
        d_zgan = d_x[self.feat_dim:]
        return score, d_zgan

    def weight_bytes(self) -> bytes:
        return (self.W1.tobytes() + self.b1.tobytes() +
                self.W2.tobytes() + self.b2.tobytes())

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1):
        """One full-batch gradient-descent step of binary cross-entropy training.
        X: (batch, feat_dim+latent_dim) inputs, y: (batch,) labels in {0,1}."""
        pre1 = X @ self.W1.T + self.b1          # (batch, hidden)
        h = np.tanh(pre1)
        pre2 = h @ self.W2.T + self.b2           # (batch, 1)
        out = sigmoid(pre2).ravel()              # (batch,)

        d_pre2 = (out - y).reshape(-1, 1)        # (batch, 1)  BCE grad wrt pre2
        d_W2 = d_pre2.T @ h / X.shape[0]
        d_b2 = d_pre2.mean(axis=0)
        d_h = d_pre2 @ self.W2                   # (batch, hidden)
        d_pre1 = d_h * (1 - h ** 2)
        d_W1 = d_pre1.T @ X / X.shape[0]
        d_b1 = d_pre1.mean(axis=0)

        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

        loss = -np.mean(y * np.log(out + 1e-9) + (1 - y) * np.log(1 - out + 1e-9))
        return loss


@dataclass
class GanGateConfig:
    latent_dim: int
    hidden_dim: int
    feat_dim: int = 32
    threshold: float = 0.5


class DesignA:
    """Gate 3 = D(feat(z), z_gan) > threshold. D is PUBLIC (stored in pk)."""

    name = "Design A (D-in-pk)"

    def __init__(self, cfg: GanGateConfig, cond_dim: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.G = Generator(cfg.latent_dim, cfg.hidden_dim, cond_dim, rng)
        self.D = Discriminator(cfg.feat_dim, cfg.latent_dim, cfg.hidden_dim, rng)

    def feat(self, z):
        return _feat(z, self.cfg.feat_dim)

    def sample_zgan(self, message: bytes, w: np.ndarray) -> np.ndarray:
        cond_rng = np.random.default_rng(int.from_bytes(hashlib.sha3_256(message).digest()[:8], "little"))
        cond = np.tanh(w[: self.G.cond_dim].astype(np.float64) / 1000.0) if self.G.cond_dim <= len(w) else \
            cond_rng.normal(0, 1, size=self.G.cond_dim)
        if len(cond) < self.G.cond_dim:
            cond = np.pad(cond, (0, self.G.cond_dim - len(cond)))
        noise = cond_rng.normal(0, 1, size=self.G.latent_dim)
        return self.G.forward(noise, cond)

    def gate3_verify(self, z: np.ndarray, z_gan: np.ndarray, **kwargs) -> bool:
        score = self.D.score(self.feat(z), z_gan)
        return score > self.cfg.threshold

    def train_discriminator(self, sample_messages: list[bytes], lattice_sampler, epochs: int = 200, lr: float = 0.2):
        """Trains D at keygen time so genuine (z, z_gan) pairs score near 1 and
        random z_gan score near 0 -- i.e. the cGAN training step described in the
        manuscript (Section IV / V.D isolation analysis)."""
        feats, zgans, labels = [], [], []
        for m in sample_messages:
            z, c, w = lattice_sampler(m)
            zg = self.sample_zgan(m, w)
            feats.append(self.feat(z)); zgans.append(zg); labels.append(1.0)
            # negative: random noise in the same latent range
            feats.append(self.feat(z)); zgans.append(self.rng.uniform(-1, 1, size=self.cfg.latent_dim)); labels.append(0.0)
        X = np.concatenate([np.stack(feats), np.stack(zgans)], axis=1)
        y = np.array(labels)
        for _ in range(epochs):
            self.D.train_step(X, y, lr=lr)

    def public_key_extra(self):
        return {"D_weights": self.D}  # explicitly exposed -- this is the vulnerability


class DesignB:
    """Gate 3 = hash-commitment check. D is discarded; only C_G is public."""

    name = "Design B (hash commitment, LGAN-II v2)"

    def __init__(self, cfg: GanGateConfig, cond_dim: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.G = Generator(cfg.latent_dim, cfg.hidden_dim, cond_dim, rng)
        self.C_G = self.G.commitment()
        # D is trained then discarded (kept here only to mirror the paper's
        # "trained during keygen, discarded" narrative -- never exposed via public_key_extra)
        self._D_discarded = Discriminator(cfg.feat_dim, cfg.latent_dim, cfg.hidden_dim, rng)
        del self._D_discarded

    def feat(self, z):
        return _feat(z, self.cfg.feat_dim)

    def sample_zgan(self, message: bytes, w: np.ndarray) -> np.ndarray:
        cond_rng = np.random.default_rng(int.from_bytes(hashlib.sha3_256(message).digest()[:8], "little"))
        cond = np.tanh(w[: self.G.cond_dim].astype(np.float64) / 1000.0) if self.G.cond_dim <= len(w) else \
            cond_rng.normal(0, 1, size=self.G.cond_dim)
        if len(cond) < self.G.cond_dim:
            cond = np.pad(cond, (0, self.G.cond_dim - len(cond)))
        noise = cond_rng.normal(0, 1, size=self.G.latent_dim)
        return self.G.forward(noise, cond)

    def sign_proof(self, z_gan: np.ndarray, w: np.ndarray, message: bytes) -> bytes:
        payload = z_gan.tobytes() + self.C_G + w.astype(np.int64).tobytes() + message
        return hashlib.sha3_256(payload).digest()

    def gate3_verify(self, z_gan: np.ndarray, w: np.ndarray, message: bytes, pi: bytes, C_G_in_sig: bytes) -> bool:
        if C_G_in_sig != self.C_G:
            return False
        payload = z_gan.tobytes() + self.C_G + w.astype(np.int64).tobytes() + message
        pi_prime = hashlib.sha3_256(payload).digest()
        return pi_prime == pi

    def public_key_extra(self):
        return {"C_G": self.C_G}  # only the commitment is public -- no differentiable target
