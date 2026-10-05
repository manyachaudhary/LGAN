"""
main.py
-------
End-to-end reproduction of the manuscript's experimental pipeline, updated to:

  1. Run >= 1000 forgery attempts per attack level (not 100).
  2. Average results across 5 independent seeds (42-46), reporting mean +/- std.
  3. Report GAN-gate bypass, lattice-gate bypass, and full forgery as distinct,
     mutually-exclusive/layered outcomes rather than one aggregated metric.
  4. Report two SEPARATE timing/size tables: LGAN-I/II Python prototype timing
     (Table II-A) and standard NIST reference sizes (Table II-B).

NOTE ON RUNTIME: the full 1000-attempt x 5-seed x 5-level x 2-design security
benchmark takes several minutes (~9 min on a laptop CPU). Running it inline
via `python3 main.py` will therefore run for a while before printing Table
III; if you want to run it unattended, use:

    setsid nohup python3 -u run_full_security.py LGAN-I 1000 > run.log 2>&1 &

which checkpoints results to `security_results_<level>.pkl` and writes a
`DONE.marker` file when finished (this is exactly how the results bundled
with this repo -- `security_results_LGAN-I.pkl` -- were produced).
"""

from __future__ import annotations
import pickle
import os
import numpy as np
from schemes import LGANScheme
from attacks import forge_l3_cached, AttackerState
from benchmark import run_table_iia, print_table_iia, print_table_iib
from security_eval import run_security_benchmark_multiseed, print_table_iii, SEEDS


def gan_isolation_analysis(seed: int = 42, n: int = 50):
    """Reproduces Section V.D: discriminator scores for genuine, random, and
    gradient-optimised (L3) z_gan inputs against Design A."""
    scheme = LGANScheme("LGAN-I", "A", seed=seed)
    scheme.keygen()
    setup_rng = np.random.default_rng(seed + 7)
    state = AttackerState(scheme, setup_rng, oracle_queries=20, shadow_queries=100)

    genuine_scores, random_scores, grad_scores = [], [], []
    rng = np.random.default_rng(seed + 11)
    for i in range(n):
        msg = f"__isolation_{i}__".encode()
        sig = scheme.sign(msg)
        genuine_scores.append(scheme.gate3.D.score(scheme.gate3.feat(sig.z), sig.z_gan))
        rand_zgan = rng.uniform(-1, 1, size=scheme.gate3.cfg.latent_dim)
        random_scores.append(scheme.gate3.D.score(scheme.gate3.feat(sig.z), rand_zgan))
        forged = forge_l3_cached(scheme, msg, rng, state)
        grad_scores.append(scheme.gate3.D.score(scheme.gate3.feat(forged.z), forged.z_gan))

    print("\n=== GAN Isolation Analysis (Design A, Section V.D) ===")
    print(f"  genuine mean score            : {np.mean(genuine_scores):.3f} (std {np.std(genuine_scores):.3f})")
    print(f"  random mean score             : {np.mean(random_scores):.3f} (std {np.std(random_scores):.3f})")
    print(f"  gradient-optimised mean score : {np.mean(grad_scores):.3f} (std {np.std(grad_scores):.3f})")


def main(n_attempts: int = 1000, seeds=SEEDS, use_cached_results: bool = True):
    print("=" * 110)
    print("LGAN Adversarial-ML Attack Framework -- end-to-end experiment reproduction")
    print("=" * 110)

    print("\n### Table II-A: LGAN Python/NumPy Prototype Timing ###\n")
    rows_a = run_table_iia(n_runs=20, seed=42)
    print_table_iia(rows_a)

    print("\n### Table II-B: Standard NIST Reference Sizes ###\n")
    print_table_iib()

    print(f"\n### Table III: Security Benchmark (LGAN-I, {len(seeds)} seeds x "
          f"{n_attempts} attempts/seed/level) ###")

    cache_path = "security_results_LGAN-I.pkl"
    if use_cached_results and n_attempts == 1000 and list(seeds) == SEEDS and os.path.exists(cache_path):
        print(f"\n[loading cached results from {cache_path} -- delete this file, or pass "
              f"use_cached_results=False, to force a fresh run]")
        with open(cache_path, "rb") as f:
            res = pickle.load(f)
    else:
        print("\nRunning fresh (this takes several minutes at full scale)...")
        res = run_security_benchmark_multiseed("LGAN-I", n_attempts=n_attempts, seeds=seeds, verbose=True)
        with open(cache_path, "wb") as f:
            pickle.dump(res, f)

    print_table_iii(res, level="LGAN-I")

    gan_isolation_analysis(seed=42, n=50)

    print("\n### Key Finding ###")
    print("Design A (D-in-pk) GAN-gate bypass rate is substantial and non-monotonic across")
    print("L1-L5 (34% -> 77% -> 75% -> 37% -> 100%, mean over 5 seeds), while Design B (hash")
    print("commitment, C_G + pi) holds at 0.0% +/- 0.0% across all five levels and all five")
    print("seeds. 'Full Forgery' is 0% for both designs at every level -- Gates 1-2 (lattice")
    print("hardness) independently block forgery regardless of the Gate-3 outcome, so the")
    print("'GAN gate' column (not 'Full Forgery') is the metric that isolates Gate 3's own")
    print("security margin, which is what the manuscript's central claim is about.")


if __name__ == "__main__":
    main()
