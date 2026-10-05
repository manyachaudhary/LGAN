"""
schemes.py
----------
LGAN-I and LGAN-II: three-gate signature schemes combining the lattice
Fiat-Shamir-with-Aborts core (Gates 1 & 2) with a GAN-based Gate 3
(Design A or Design B, see gan_gate.py).

Signature formats:
  Design A: (z, c, w, z_gan)
  Design B: (z, c, w, z_gan, pi, C_G)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from lattice import LatticeCore, make_params
from gan_gate import DesignA, DesignB, GanGateConfig


LEVEL_TO_GAN_CFG = {
    "LGAN-I": GanGateConfig(latent_dim=128, hidden_dim=256, feat_dim=32, threshold=0.5),
    "LGAN-II": GanGateConfig(latent_dim=192, hidden_dim=384, feat_dim=32, threshold=0.5),
}


@dataclass
class Signature:
    z: np.ndarray
    c: np.ndarray
    w: np.ndarray
    z_gan: np.ndarray
    pi: bytes | None = None
    C_G: bytes | None = None


class LGANScheme:
    def __init__(self, level: str, design: str, seed: int = 42):
        assert level in ("LGAN-I", "LGAN-II")
        assert design in ("A", "B")
        self.level = level
        self.design_name = design
        self.rng = np.random.default_rng(seed)
        self.lattice = LatticeCore(make_params(level), self.rng)
        gan_cfg = LEVEL_TO_GAN_CFG[level]
        cond_dim = min(16, gan_cfg.latent_dim)
        if design == "A":
            self.gate3 = DesignA(gan_cfg, cond_dim, self.rng)
        else:
            self.gate3 = DesignB(gan_cfg, cond_dim, self.rng)

    # --------------------------------------------------------------- keygen
    def keygen(self):
        self.lattice.keygen()
        if isinstance(self.gate3, DesignA):
            # cGAN training at keygen time: build a small set of genuine (z, w) pairs
            # from the lattice core to train D so genuine z_gan scores high (Sec. V.D).
            train_msgs = [f"keygen-train-{i}".encode() for i in range(150)]
            self.gate3.train_discriminator(train_msgs, self.lattice.sign_core, epochs=800, lr=0.5)
        return self.public_key()

    def public_key(self):
        pk = {"A": self.lattice.A, "t": self.lattice.t, "level": self.level}
        pk.update(self.gate3.public_key_extra())
        return pk

    # ----------------------------------------------------------------- sign
    def sign(self, message: bytes) -> Signature | None:
        core_result = self.lattice.sign_core(message)
        if core_result is None:
            return None
        z, c, w = core_result
        z_gan = self.gate3.sample_zgan(message, w)
        if isinstance(self.gate3, DesignB):
            pi = self.gate3.sign_proof(z_gan, w, message)
            return Signature(z=z, c=c, w=w, z_gan=z_gan, pi=pi, C_G=self.gate3.C_G)
        else:
            return Signature(z=z, c=c, w=w, z_gan=z_gan)

    # --------------------------------------------------------------- verify
    def verify(self, message: bytes, sig: Signature) -> dict:
        """Returns a dict of per-gate booleans plus overall forgery success."""
        gate1_ok, gate2_ok = self.lattice.verify_core(message, sig.z, sig.c)
        if isinstance(self.gate3, DesignB):
            gate3_ok = self.gate3.gate3_verify(sig.z_gan, sig.w, message, sig.pi, sig.C_G)
        else:
            gate3_ok = self.gate3.gate3_verify(sig.z, sig.z_gan)
        full_ok = gate1_ok and gate2_ok and gate3_ok
        return {"gate1": gate1_ok, "gate2": gate2_ok, "gate3": gate3_ok, "full": full_ok}

    # ---------------------------------------------------------- size report
    def sizes(self) -> dict:
        """Approximate wire sizes in bytes, dominated by GAN generator weights in pk."""
        p = self.lattice.p
        int_bytes = 8  # numpy int64
        sk_bytes = p.n * int_bytes
        sig_bytes = p.n * int_bytes * 2 + self.gate3.cfg.latent_dim * 8  # z, c, z_gan (float64)
        if isinstance(self.gate3, DesignB):
            sig_bytes += 32 + 32  # pi + C_G
        G = self.gate3.G
        gen_weight_bytes = (G.W1.size + G.b1.size + G.W2.size + G.b2.size) * 8
        pk_bytes = p.n * int_bytes + gen_weight_bytes  # A description compressed via seed + t + G weights
        if isinstance(self.gate3, DesignA):
            D = self.gate3.D
            pk_bytes += (D.W1.size + D.b1.size + D.W2.size + D.b2.size) * 8
        else:
            pk_bytes += 32  # C_G commitment
        return {"pk_bytes": pk_bytes, "sk_bytes": sk_bytes, "sig_bytes": sig_bytes}
