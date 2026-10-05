# LGAN Attack Framework — Runnable Reproduction

This is an end-to-end, runnable implementation of the experiment described in
*"Adversarial Machine Learning Threat Analysis for GAN-Augmented Post-Quantum
Lattice-Based Signature Schemes: A Five-Level Attack Framework"* (Faridi,
Chaudhary, Masood).

It is a **from-scratch reference implementation in Python/NumPy**, matching the
paper's own stated methodology. It is not constant-time, not side-channel-
hardened, and not meant for production use — it exists to make the paper's
empirical claims runnable and inspectable.

## Files

| File | Contents |
|---|---|
| `lattice.py` | Fiat-Shamir-with-Aborts lattice signature core (Gates 1 & 2). |
| `gan_gate.py` | cGAN Generator/Discriminator (NumPy MLPs, hand-derived backprop), and the two Gate-3 designs: `DesignA` (discriminator in `pk`) and `DesignB` (hash commitment `C_G = SHA3-256(G.weights)` + signing proof `pi`). |
| `schemes.py` | `LGANScheme`: wires the lattice core + a chosen Gate-3 design into LGAN-I / LGAN-II. |
| `reference_schemes.py` | Toy Dilithium2 / Falcon-512 / SPHINCS+-SHA2-128f implementations, used to source spec-exact size constants and (optionally) illustrative same-process timing. |
| `attacks.py` | L1-L5 attack framework, plus `AttackerState` (one-time oracle/shadow-model setup reused across all forgery attempts) and the `CACHED_ATTACKS` dict used by the benchmark. |
| `security_eval.py` | **Table III**: multi-seed, large-sample, layered-outcome security benchmark (see "What changed" below). |
| `run_full_security.py` | Standalone script to run the full-scale security benchmark (takes several minutes) and checkpoint results to `security_results_<level>.pkl` + a `DONE.marker` file. |
| `benchmark.py` | **Table II-A** (LGAN Python/NumPy prototype timing) and **Table II-B** (standard NIST reference sizes), now reported as two separate tables. |
| `main.py` | Runs everything: Table II-A, Table II-B, Table III (loading the cached `.pkl` if present, else running fresh), and the Section V.D GAN-isolation analysis. |
| `make_figure.py` | Renders `fig1_reproduced.png` with error bars from the multi-seed results. |
| `security_results_LGAN-I.pkl` | Pre-computed full-scale results (5 seeds x 1000 attempts x 5 levels x 2 designs) bundled so `main.py` and `make_figure.py` can render Table III/Fig. 1 instantly without a multi-minute rerun. Delete it (or pass `use_cached_results=False` to `main()`) to force a fresh run. |


## Running it

```bash
pip install numpy matplotlib   # matplotlib only needed for make_figure.py

# Fast path: Table II-A, Table II-B, and Table III from the bundled
# 5-seed x 1000-attempt results (seconds):
python3 main.py

# Force a full fresh run of the security benchmark (several minutes):
python3 -c "from security_eval import run_security_benchmark_multiseed; \
            run_security_benchmark_multiseed('LGAN-I', n_attempts=1000)"

# Or run it in the background (recommended -- ~9 min at full scale) and
# check on it later:
setsid nohup python3 -u run_full_security.py LGAN-I 1000 > run.log 2>&1 &
tail -f run.log            # watch progress
ls DONE.marker             # appears when finished

python3 make_figure.py     # renders fig1_reproduced.png (error bars included)
```

## Sample Table III (bundled results: LGAN-I, 5 seeds × 1000 attempts/level)

```
Attack  Design  No bypass       Lattice only    GAN only        Full forgery    Lattice gate    GAN gate
--------------------------------------------------------------------------------------------------------
L1      A        65.9( 0.7)       0.0( 0.0)      34.1( 0.7)       0.0( 0.0)       0.0( 0.0)      34.1( 0.7)
L1      B       100.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)
L2      A        23.2( 6.4)       0.0( 0.0)      76.8( 6.4)       0.0( 0.0)       0.0( 0.0)      76.8( 6.4)
L2      B       100.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)
L3      A        25.4( 0.8)       0.0( 0.0)      74.6( 0.8)       0.0( 0.0)       0.0( 0.0)      74.6( 0.8)
L3      B       100.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)
L4      A        62.6( 2.1)       0.0( 0.0)      37.4( 2.1)       0.0( 0.0)       0.0( 0.0)      37.4( 2.1)
L4      B       100.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)
L5      A         0.1( 0.2)       0.0( 0.0)      99.9( 0.2)       0.0( 0.0)       0.0( 0.0)      99.9( 0.2)
L5      B       100.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)       0.0( 0.0)
```
(values are mean% with std% across seeds in parentheses; pooled n=5,000 per cell)

Design B's worst-case (L5) GAN-gate pooled Wilson 95% upper bound over all
5,000 pooled attempts is **0.08%**.

## What the results show

* **Table II-A/B** — LGAN public keys are dominated by generator weights
  (hundreds of KB) vs. sub-2KB NIST public keys; the NIST reference sizes in
  Table II-B are exact published spec values (Dilithium2 pk=1312B/
  sk=2528B/sig=2420B, Falcon-512 pk=897B/sk=1281B/sig=666B, SPHINCS+-128f
  pk=32B/sk=64B/sig=17,088B).
* **Table III** — Design A's GAN-gate bypass rate is substantial and
  non-monotonic across attack levels (naive random 34% → oracle 77% →
  gradient ascent 75% → surrogate transfer 37% → combined 100%, means over
  5 seeds), while Design B holds at **0.0% ± 0.0%** across all five levels and
  all five seeds. `full_forgery` is 0% for both designs at every level and
  seed, because Gates 1–2 (lattice hardness) independently block forgery
  regardless of Gate 3 — exactly as the paper explains, `gan_gate_pass_rate`
  (not `full_forgery`) is the metric that isolates Gate 3's own security
  margin, which is what the manuscript's central claim is about.
* **Section V.D isolation analysis** — genuine z_gan scores high under D,
  uniform-random z_gan scores markedly lower, and gradient-ascent-optimised
  z_gan closes much of that gap — reproducing the paper's qualitative finding
  that a public differentiable discriminator provides little security margin
  once an attacker can run gradient ascent.
