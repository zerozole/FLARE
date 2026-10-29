"""Redraw Figure 3 (Qwen2.5-VL-7b five-class confusion matrices, base and QLoRA) from its counts.
The old figs/qwen_*_confusion.pdf were crops of the OJAp preprint pages and left marks at the crop edge.
Counts are those printed in the old figures (synthetic test split by curve, 4,628 curves).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["non_flare", "spike", "gaussian", "gamma", "fred"]
MATS = {
    "qwen_base_confusion_v2.pdf": [[900, 2, 13, 11, 0], [911, 5, 2, 7, 0], [905, 9, 5, 7, 0],
                                   [913, 5, 2, 5, 0], [914, 2, 2, 8, 0]],
    "qwen_qlora_confusion_v2.pdf": [[722, 3, 1, 200, 0], [828, 8, 1, 88, 0], [791, 16, 0, 119, 0],
                                    [794, 8, 0, 123, 0], [759, 6, 0, 161, 0]],
}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
for name, m in MATS.items():
    m = np.array(m)
    fig, ax = plt.subplots(figsize=(3.0, 2.55))
    im = ax.imshow(m, cmap="Blues", vmin=0)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.5 * m.max() else "black")
    ax.set_xticks(range(5), LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(5), LABELS, fontsize=7.5)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(str(_P(__file__).with_name(name)), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name, m.sum(), "curves")
