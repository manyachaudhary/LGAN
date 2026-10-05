"""
make_figure.py
--------------
Renders a bar chart (with error bars across the 5 seeds) reproducing Fig. 1 of
the manuscript: GAN-gate bypass rate across L1-L5 for Design A vs Design B.

Uses the cached security_results_LGAN-I.pkl if present (produced by
run_full_security.py); otherwise runs a fresh multi-seed benchmark.
"""
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from security_eval import run_security_benchmark_multiseed, SEEDS

cache_path = "security_results_LGAN-I.pkl"
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        res = pickle.load(f)
else:
    res = run_security_benchmark_multiseed("LGAN-I", n_attempts=1000, seeds=SEEDS, verbose=False)

summ = res["summary"]
levels = ["L1", "L2", "L3", "L4", "L5"]
a_mean = [summ["Design A"][l]["gan_gate_pass_rate"]["mean"] * 100 for l in levels]
a_std = [summ["Design A"][l]["gan_gate_pass_rate"]["std"] * 100 for l in levels]
b_mean = [summ["Design B"][l]["gan_gate_pass_rate"]["mean"] * 100 for l in levels]
b_std = [summ["Design B"][l]["gan_gate_pass_rate"]["std"] * 100 for l in levels]

x = np.arange(len(levels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_a = ax.bar(x - width / 2, a_mean, width, yerr=a_std, capsize=4,
                 label="Design A (D-in-pk)", color="#d9534f")
bars_b = ax.bar(x + width / 2, b_mean, width, yerr=b_std, capsize=4,
                 label="Design B (hash commitment, LGAN-II v2)", color="#5cb85c")

ax.set_ylabel("GAN gate bypass rate (%)")
ax.set_xlabel("Attack Level")
ax.set_title("Five-Level Adversarial Attack Framework: GAN Gate Bypass Rate\n"
              "(mean \u00b1 std over 5 seeds, LGAN-I, 1000 attempts/seed/level)")
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars, means in ((bars_a, a_mean), (bars_b, b_mean)):
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("fig1_reproduced.png", dpi=150)
print("Saved fig1_reproduced.png")
