"""
security_eval.py
-----------------
Reproduces Table III of the manuscript, with four additions requested on top
of the original single-seed / 100-attempt version:

  1. Scale: >= 1000 forgery attempts per attack level (default 1000).
  2. Multi-seed averaging: independent runs across seeds 42-46 (5 seeds),
     reporting mean +/- standard deviation for every metric.
  3. Decoupled gate outcomes: rather than a single aggregated "success" flag,
     each forgery attempt is classified into one of four MUTUALLY EXCLUSIVE
     outcome buckets:
         - no_bypass          : lattice gates fail AND GAN gate fails
         - lattice_only       : lattice gates pass, GAN gate fails
         - gan_only           : GAN gate passes, lattice gates fail
         - full_forgery       : lattice gates pass AND GAN gate passes
     plus the two MARGINAL (non-exclusive) rates that were reported before:
     lattice_gate_pass_rate = P(gate1 AND gate2) and gan_gate_pass_rate =
     P(gate3), which are the layered building blocks the four buckets above
     are constructed from (marginal = lattice_only + full_forgery, and
     gan_gate_pass_rate = gan_only + full_forgery).
  4. (Timing tables moved to benchmark.py -- see that file for the two
     separated Table II variants.)
"""

from __future__ import annotations
import time
import numpy as np
from schemes import LGANScheme
from attacks import CACHED_ATTACKS, AttackerState

SEEDS = [42, 43, 44, 45, 46]
OUTCOME_KEYS = ["no_bypass", "lattice_only", "gan_only", "full_forgery"]


def wilson_upper_bound(successes: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score interval upper bound for a binomial proportion."""
    if n == 0:
        return 0.0
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    centre = p_hat + z ** 2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    return (centre + margin) / denom


def _classify(res: dict) -> str:
    """Maps a scheme.verify() result dict to one of the four mutually
    exclusive, layered outcome buckets."""
    lattice_ok = res["gate1"] and res["gate2"]
    gan_ok = res["gate3"]
    if lattice_ok and gan_ok:
        return "full_forgery"
    if lattice_ok and not gan_ok:
        return "lattice_only"
    if gan_ok and not lattice_ok:
        return "gan_only"
    return "no_bypass"


def _run_single_seed(level: str, design: str, attack_name: str, attack_fn,
                      n_attempts: int, seed: int) -> dict:
    """Runs one (design, attack_level) combination for one seed and returns
    raw counts for all four outcome buckets plus the two marginal gate rates."""
    scheme = LGANScheme(level, design, seed=seed)
    scheme.keygen()
    setup_rng = np.random.default_rng(seed + 999)
    state = AttackerState(scheme, setup_rng, oracle_queries=50, shadow_queries=400)

    rng = np.random.default_rng(seed + (hash(attack_name) % 100000))
    counts = {k: 0 for k in OUTCOME_KEYS}
    lattice_hits, gan_hits = 0, 0
    for i in range(n_attempts):
        msg = f"__target_message_{i}__".encode()
        forged = attack_fn(scheme, msg, rng, state)
        res = scheme.verify(msg, forged)
        bucket = _classify(res)
        counts[bucket] += 1
        lattice_hits += int(res["gate1"] and res["gate2"])
        gan_hits += int(res["gate3"])

    return {
        "counts": counts,
        "lattice_gate_hits": lattice_hits,
        "gan_gate_hits": gan_hits,
        "n": n_attempts,
    }


def run_security_benchmark_multiseed(level: str = "LGAN-I", n_attempts: int = 1000,
                                      seeds: list = SEEDS, verbose: bool = True) -> dict:
    """Runs every (design x attack level) combination across all `seeds`,
    returning per-seed raw results plus aggregated mean/std summary stats."""
    results = {"per_seed": {}, "summary": {}}
    for design in ["A", "B"]:
        results["summary"][f"Design {design}"] = {}
        for attack_name, attack_fn in CACHED_ATTACKS.items():
            seed_rates = {k: [] for k in OUTCOME_KEYS}
            seed_rates["lattice_gate_pass_rate"] = []
            seed_rates["gan_gate_pass_rate"] = []
            total_gan_hits, total_n = 0, 0

            for seed in seeds:
                t0 = time.time()
                raw = _run_single_seed(level, design, attack_name, attack_fn, n_attempts, seed)
                results["per_seed"][(level, design, attack_name, seed)] = raw
                n = raw["n"]
                for k in OUTCOME_KEYS:
                    seed_rates[k].append(raw["counts"][k] / n)
                seed_rates["lattice_gate_pass_rate"].append(raw["lattice_gate_hits"] / n)
                seed_rates["gan_gate_pass_rate"].append(raw["gan_gate_hits"] / n)
                total_gan_hits += raw["gan_gate_hits"]
                total_n += n
                if verbose:
                    print(f"    seed={seed} [{level}/Design {design}/{attack_name}] "
                          f"gan_gate={raw['gan_gate_hits']}/{n} "
                          f"full={raw['counts']['full_forgery']}/{n} "
                          f"({time.time()-t0:.1f}s)")

            summary = {}
            for k, vals in seed_rates.items():
                arr = np.array(vals)
                summary[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
            summary["gan_gate_wilson_upper_95_pooled"] = wilson_upper_bound(total_gan_hits, total_n)
            summary["pooled_n"] = total_n
            results["summary"][f"Design {design}"][attack_name] = summary
            if verbose:
                gm = summary["gan_gate_pass_rate"]
                fm = summary["full_forgery"]
                print(f"  >> {level}/Design {design}/{attack_name}: "
                      f"GAN gate = {gm['mean']*100:.1f}% +/- {gm['std']*100:.1f}%  |  "
                      f"Full forgery = {fm['mean']*100:.1f}% +/- {fm['std']*100:.1f}%  "
                      f"(pooled n={total_n})")
    return results


def print_table_iii(results: dict, level: str = "LGAN-I"):
    summ = results["summary"]
    n_per_seed = results["per_seed"][(level, "A", "L1", SEEDS[0])]["n"]
    print(f"\nTable III -- Security Benchmark ({level}, {len(SEEDS)} seeds x "
          f"{n_per_seed} attempts/seed/level)")
    print("Layered / mutually-exclusive outcome rates: mean% (std%) across seeds\n")
    col_w = 16
    header = (f"{'Attack':<8}{'Design':<8}"
              f"{'No bypass':<{col_w}}{'Lattice only':<{col_w}}{'GAN only':<{col_w}}"
              f"{'Full forgery':<{col_w}}{'Lattice gate':<{col_w}}{'GAN gate':<{col_w}}")
    print(header)
    print("-" * len(header))
    for level_name in ["L1", "L2", "L3", "L4", "L5"]:
        for design in ["A", "B"]:
            s = summ[f"Design {design}"][level_name]

            def fmt(key):
                m, sd = s[key]["mean"] * 100, s[key]["std"] * 100
                return f"{m:5.1f}({sd:4.1f})"

            print(f"{level_name:<8}{design:<8}"
                  f"{fmt('no_bypass'):<{col_w}}{fmt('lattice_only'):<{col_w}}"
                  f"{fmt('gan_only'):<{col_w}}{fmt('full_forgery'):<{col_w}}"
                  f"{fmt('lattice_gate_pass_rate'):<{col_w}}{fmt('gan_gate_pass_rate'):<{col_w}}")

    print("\nNotes:")
    print("  - 'No bypass'/'Lattice only'/'GAN only'/'Full forgery' are MUTUALLY EXCLUSIVE")
    print("    and sum to 100% per row (up to rounding).")
    print("  - 'Lattice gate' = P(gate1 AND gate2) and 'GAN gate' = P(gate3) are the two")
    print("    MARGINAL layer-pass rates (lattice_only + full_forgery, and gan_only + full_forgery")
    print("    respectively) -- reported for direct comparability with the single aggregated")
    print("    metric used in the original 100-attempt/single-seed benchmark.")
    b_l5 = summ["Design B"]["L5"]
    print(f"\n  Design B worst-case (L5) GAN-gate pooled Wilson 95% upper bound "
          f"(pooled n={b_l5['pooled_n']}): {b_l5['gan_gate_wilson_upper_95_pooled']*100:.2f}%")


if __name__ == "__main__":
    res = run_security_benchmark_multiseed("LGAN-I", n_attempts=1000, seeds=SEEDS)
    print_table_iii(res, level="LGAN-I")
