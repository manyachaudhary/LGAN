"""
benchmark.py
------------
Reproduces Table II of the manuscript, restructured into two SEPARATE tables
as requested:

  Table II-A -- LGAN-I / LGAN-II Python/NumPy prototype timings (sign/verify
                median ms over N runs, keygen time, and this prototype's own
                pk/sk/sig sizes). This is a same-process, same-hardware,
                like-for-like comparison between LGAN-I and LGAN-II only.

  Table II-B -- Standard NIST reference sizes (Dilithium2/ML-DSA-44,
                Falcon-512, SPHINCS+-SHA2-128f): pk/sk/signature sizes taken
                directly from the published FIPS 204/205/206 specifications
                (not measured from this benchmark's toy implementations).
                No timing numbers are reported in this table, since a Python
                prototype's wall-clock time is not a meaningful stand-in for
                the optimised C reference implementations' performance -- the
                manuscript itself flags this gap explicitly.

`reference_schemes.py` (toy Dilithium2/Falcon-512/SPHINCS+ implementations)
is still available for anyone who wants an in-process, same-hardware Python
timing comparison; see `bench_toy_reference_schemes()` below if that is
useful, but it is deliberately NOT part of either headline table.
"""

from __future__ import annotations
import time
import numpy as np
from schemes import LGANScheme
from reference_schemes import Dilithium2Toy, Falcon512Toy, SphincsPlus128fToy


def _median_ms(times):
    return float(np.median(times)) * 1000.0


def _std_ms(times):
    return float(np.std(times)) * 1000.0


# --------------------------------------------------------------- Table II-A
def bench_lgan(level: str, design: str = "B", n_runs: int = 20, seed: int = 42):
    scheme = LGANScheme(level, design, seed=seed)
    t0 = time.perf_counter()
    scheme.keygen()
    keygen_s = time.perf_counter() - t0

    sign_times, verify_times = [], []
    sigs = []
    for i in range(n_runs):
        msg = f"__bench_msg_{i}__".encode()
        t0 = time.perf_counter()
        sig = scheme.sign(msg)
        sign_times.append(time.perf_counter() - t0)
        sigs.append((msg, sig))
    for msg, sig in sigs:
        t0 = time.perf_counter()
        scheme.verify(msg, sig)
        verify_times.append(time.perf_counter() - t0)

    sizes = scheme.sizes()
    return {
        "scheme": f"{level} ({scheme.gate3.name})",
        "keygen_s": keygen_s,
        "sign_ms_median": _median_ms(sign_times),
        "sign_ms_std": _std_ms(sign_times),
        "verify_ms_median": _median_ms(verify_times),
        "verify_ms_std": _std_ms(verify_times),
        "pk_KB": sizes["pk_bytes"] / 1024,
        "sk_KB": sizes["sk_bytes"] / 1024,
        "sig_KB": sizes["sig_bytes"] / 1024,
    }


def run_table_iia(n_runs: int = 20, seed: int = 42):
    """LGAN-I / LGAN-II Python/NumPy prototype timings only."""
    return [
        bench_lgan("LGAN-I", "B", n_runs, seed),
        bench_lgan("LGAN-II", "B", n_runs, seed),
    ]


def print_table_iia(rows):
    print("Table II-A -- LGAN Python/NumPy Prototype Timing (this benchmark, same process/hardware)")
    print(f"{'Scheme':<38}{'Keygen(s)':>10}{'Sign(ms)':>12}{'Verify(ms)':>14}{'pk(KB)':>9}{'sk(KB)':>9}{'sig(KB)':>9}")
    print("-" * 101)
    for r in rows:
        print(f"{r['scheme']:<38}{r['keygen_s']:>10.3f}"
              f"{r['sign_ms_median']:>8.3f}\u00b1{r['sign_ms_std']:<3.2f}"
              f"{r['verify_ms_median']:>9.4f}\u00b1{r['verify_ms_std']:<3.3f}"
              f"{r['pk_KB']:>9.2f}{r['sk_KB']:>9.2f}{r['sig_KB']:>9.2f}")
    print("(Sign/Verify columns: median\u00b1std over N=%d runs.)\n" % 20)


# --------------------------------------------------------------- Table II-B
# Standard, published NIST reference sizes (FIPS 204 / FIPS 205 / FIPS 206).
# These are specification constants, not benchmark measurements.
NIST_REFERENCE_SIZES = [
    {"scheme": "Dilithium2 / ML-DSA-44 (FIPS 204)", "security_level": "NIST L2",
     "pk_bytes": Dilithium2Toy.PK_BYTES, "sk_bytes": Dilithium2Toy.SK_BYTES, "sig_bytes": Dilithium2Toy.SIG_BYTES},
    {"scheme": "Falcon-512 / FN-DSA-512 (FIPS 206, draft)", "security_level": "NIST L1",
     "pk_bytes": Falcon512Toy.PK_BYTES, "sk_bytes": Falcon512Toy.SK_BYTES, "sig_bytes": Falcon512Toy.SIG_BYTES},
    {"scheme": "SPHINCS+-SHA2-128f (FIPS 205)", "security_level": "NIST L1",
     "pk_bytes": SphincsPlus128fToy.PK_BYTES, "sk_bytes": SphincsPlus128fToy.SK_BYTES,
     "sig_bytes": SphincsPlus128fToy.SIG_BYTES},
]


def print_table_iib(rows=NIST_REFERENCE_SIZES):
    print("Table II-B -- Standard NIST Reference Key/Signature Sizes (published spec values,")
    print("              not benchmark measurements; no timing reported here -- see note below)")
    print(f"{'Scheme':<42}{'Level':<10}{'pk (B)':>9}{'sk (B)':>9}{'sig (B)':>10}{'pk (KB)':>10}{'sig (KB)':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['scheme']:<42}{r['security_level']:<10}{r['pk_bytes']:>9}{r['sk_bytes']:>9}"
              f"{r['sig_bytes']:>10}{r['pk_bytes']/1024:>10.2f}{r['sig_bytes']/1024:>10.2f}")
    print("\nNote: absolute Python-prototype wall-clock timings for these NIST schemes are not")
    print("reported here because they are not comparable to the optimised C reference")
    print("implementations' performance (the manuscript notes a ~50x gap). Sizes above are")
    print("implementation-independent and match the published FIPS 204/205/206 specifications")
    print("exactly, which is the only NIST-comparison claim Table II-B makes.")


# ---------------------------------------------------- optional: toy timing
def bench_toy_reference_schemes(n_runs: int = 20, seed: int = 42):
    """OPTIONAL same-process Python/NumPy timing for the toy NIST reference
    implementations in reference_schemes.py. Not part of Table II-A or II-B;
    provided only for readers who want an (illustrative, not authoritative)
    in-process timing comparison. See reference_schemes.py's module docstring
    for the caveats on why this should not be read as a real NIST comparison."""
    rows = []

    scheme = Dilithium2Toy(seed=seed)
    scheme.keygen()
    st, vt = [], []
    sigs = []
    for i in range(n_runs):
        msg = f"__bench_msg_{i}__".encode()
        t0 = time.perf_counter(); z, c, w = scheme.sign(msg); st.append(time.perf_counter() - t0)
        sigs.append((msg, z, c, w))
    for msg, z, c, w in sigs:
        t0 = time.perf_counter(); scheme.verify(msg, z, c, w); vt.append(time.perf_counter() - t0)
    rows.append({"scheme": "Dilithium2 [toy, this process]", "sign_ms": _median_ms(st), "verify_ms": _median_ms(vt)})

    scheme = Falcon512Toy(seed=seed + 1)
    scheme.keygen()
    st, vt = [], []
    sigs = []
    for i in range(n_runs):
        msg = f"__bench_msg_{i}__".encode()
        t0 = time.perf_counter(); s, digest = scheme.sign(msg); st.append(time.perf_counter() - t0)
        sigs.append((msg, s, digest))
    for msg, s, digest in sigs:
        t0 = time.perf_counter(); scheme.verify(msg, s, digest); vt.append(time.perf_counter() - t0)
    rows.append({"scheme": "Falcon-512 [toy, this process]", "sign_ms": _median_ms(st), "verify_ms": _median_ms(vt)})

    scheme = SphincsPlus128fToy(seed=seed + 2)
    scheme.keygen()
    st, vt = [], []
    sigs = []
    for i in range(n_runs):
        msg = f"__bench_msg_{i}__".encode()
        t0 = time.perf_counter(); fs, ws = scheme.sign(msg); st.append(time.perf_counter() - t0)
        sigs.append((msg, fs, ws))
    for msg, fs, ws in sigs:
        t0 = time.perf_counter(); scheme.verify(msg, fs, ws); vt.append(time.perf_counter() - t0)
    rows.append({"scheme": "SPHINCS+-128f [toy, this process]", "sign_ms": _median_ms(st), "verify_ms": _median_ms(vt)})

    return rows


if __name__ == "__main__":
    rows_a = run_table_iia(n_runs=20, seed=42)
    print_table_iia(rows_a)
    print()
    print_table_iib()
