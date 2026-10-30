"""Remake Figure 2b (binary precision vs recall on the synthetic test split) from the saved predictions,
ml4ps/bench_metrics.csv (written by bench_metrics.py), with the model names used in the appendix table.
The old figs/vlm_precision_recall.png had hand-placed points (Grok, Mistral and GPT-5 recall off by 1-2.5 pts).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = str(_P(__file__).resolve().parent.parent / "bench_metrics.csv")
OUT = str(_P(__file__).with_name("vlm_precision_recall_v2.pdf"))

# legend order and colours (distinct hues; the QLoRA model is hollow to pair it with its base model)
STYLE = {
    "GPT-5":                  "#e377c2",
    "GPT-5 mini":             "#d62728",
    "GPT-5-nano":             "#1f77b4",
    "Claude 3 Haiku":         "#9467bd",
    "Grok-4.1-fast":          "#2ca02c",
    "Qwen-3.5-plus":          "#bcbd22",
    "Qwen3-VL-235B-A22B":     "#17becf",
    "Qwen3-VL-30B-A3B":       "#333333",
    "Mistral Large 3":        "#7f7f7f",
    "Kimi-k2.5":              "#8c564b",
    "Qwen2.5-VL-7b-Instruct": "#ff7f0e",
    "Qwen2.5-VL-7b-QLoRA":    "#ff7f0e",
}

rows = {r["model"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for name, colour in STYLE.items():
    r = rows[name]
    x, y = 100 * float(r["binary_flare_recall"]), 100 * float(r["binary_flare_precision"])
    hollow = name.endswith("QLoRA")
    ax.scatter(x, y, s=90, color="white" if hollow else colour, edgecolors=colour,
               linewidths=2 if hollow else 0.5, label=name, zorder=3)

ax.set_xlim(-2, 82)
ax.set_ylim(38, 95)
ax.set_xlabel("Binary recall (%)", fontsize=12)
ax.set_ylabel("Binary precision (%)", fontsize=12)
ax.grid(True, ls="--", lw=0.5, alpha=0.4)
ax.tick_params(direction="in", top=True, right=True, labelsize=10)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True, borderaxespad=0)
fig.subplots_adjust(left=0.10, right=0.66, top=0.97, bottom=0.14)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT)
