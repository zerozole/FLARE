"""Build the three-panel rendering figure for the camera-ready paper.

All panels are the exact PNGs the models see: (a) a simulated gaussian-flare curve from the
training corpus and (b) real candidate 2014051, both without axes (CNN/ViT renders), and
(c) the same candidate with axes, as sent to the VLMs (plots_candidates/2014051.png).
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SIM = str(DATA_ROOT / "000074.png")
REAL = str(DATA_ROOT / "2014051.png")
REAL_AXES = str(DATA_ROOT / "plots_candidates/2014051.png")
OUT = str(_P(__file__).with_name("sim_vs_real_axes.pdf"))

ax_img = mpimg.imread(REAL_AXES)
aspect = ax_img.shape[1] / ax_img.shape[0]
fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.95), gridspec_kw={"width_ratios": [1, 1, aspect]})

for ax, img, label in [
    (axes[0], mpimg.imread(SIM), "(a) Simulated, no axes"),
    (axes[1], mpimg.imread(REAL), "(b) Real, no axes"),
    (axes[2], ax_img, "(c) Real, with axes"),
]:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.75")
        spine.set_linewidth(0.6)
    ax.set_xlabel(label, fontsize=8, labelpad=3)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.12, wspace=0.05)
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
