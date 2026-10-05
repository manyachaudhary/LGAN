"""
run_full_security.py
---------------------
Standalone runner for the full-scale (1000 attempts x 5 seeds) security
benchmark. Designed to be launched in the background since it takes several
minutes; writes a completion marker file when done.
"""
import pickle
import sys
from security_eval import run_security_benchmark_multiseed, print_table_iii, SEEDS

if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "LGAN-I"
    n_attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Starting full security benchmark: level={level}, n_attempts={n_attempts}, seeds={SEEDS}", flush=True)
    res = run_security_benchmark_multiseed(level, n_attempts=n_attempts, seeds=SEEDS, verbose=True)
    print_table_iii(res, level=level)

    with open(f"security_results_{level}.pkl", "wb") as f:
        pickle.dump(res, f)

    with open("DONE.marker", "w") as f:
        f.write("done\n")
    print("\n[run_full_security.py] COMPLETE", flush=True)
