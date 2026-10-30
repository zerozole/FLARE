"""Remake Figure 2a, the recognition engine, as a vector flowchart that follows run_engine.py exactly:
round 1, both classifiers label every curve; agreed curves are final; disputed curves go to the
evaluator, which also writes feedback; round 2, the classifiers relabel only the disputed curves with that
feedback; curves still disputed take the evaluator's verdict.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = str(_P(__file__).with_name("vlm_engine_arch_v2.pdf"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
fig, ax = plt.subplots(figsize=(3.9, 2.9))
ax.set_xlim(-1.0, 10.1)
ax.set_ylim(2.45, 10.25)
ax.axis("off")

C_IN, C_CLS, C_EV, C_OUT = "#fde2e2", "#e3e6fb", "#efe1f2", "#dcf3dc"


def box(x, y, w, h, title, sub, colour):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=colour, ec="0.25", lw=0.8))
    ax.text(x, y + (0.2 if sub else 0), title, ha="center", va="center", fontsize=9, weight="bold")
    if sub:
        ax.text(x, y - 0.27, sub, ha="center", va="center", fontsize=7.5)


def arrow(x0, y0, x1, y1, ls="-", rad=0.0, text=None, tx=0, ty=0, ha="center"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2", ls=ls,
                                                        shrinkA=0, shrinkB=0, mutation_scale=8,
                                                        connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=7.5, style="italic")


# input
box(5, 9.7, 5.2, 0.9, "Light-curve image", "rendered candidate", C_IN)
# classifiers
box(2.4, 7.75, 3.9, 1.0, "Classifier A", "Grok-4.1-fast, high recall", C_CLS)
box(7.6, 7.75, 3.9, 1.0, "Classifier B", "Qwen3.5-plus, high precision", C_CLS)
arrow(4.0, 9.25, 2.6, 8.25)
arrow(6.0, 9.25, 7.4, 8.25)
# agreement test
dx, dy, cy = 1.2, 0.6, 5.95
ax.add_patch(Polygon([(5 - dx, cy), (5, cy + dy), (5 + dx, cy), (5, cy - dy)], closed=True,
                     fc="white", ec="0.25", lw=0.8))
ax.text(5, cy, "Agree?", ha="center", va="center", fontsize=8.5, weight="bold")
arrow(2.4, 7.25, 4.25, 6.3)
arrow(7.6, 7.25, 5.75, 6.3)
# evaluator (left) and final label (right)
box(2.2, 3.7, 3.8, 1.0, "Evaluator", "GPT-5, adjudicates", C_EV)
box(7.8, 3.7, 3.6, 1.0, "Final label", "flare / non-flare", C_OUT)
arrow(5 + dx, cy, 7.8, 4.2, text="yes", tx=7.05, ty=5.35)
arrow(5 - dx, cy, 2.2, 4.2, text="no", tx=2.95, ty=5.35)
# feedback loop: evaluator back to the classifiers for round 2
ax.plot([0.3, -0.35, -0.35], [3.7, 3.7, 7.75], color="0.2", lw=0.8, ls="--")
ax.annotate("", (0.45, 7.75), (-0.35, 7.75), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2", ls="--",
                                                             mutation_scale=8, shrinkA=0, shrinkB=0))
ax.text(-0.72, 5.72, "feedback to A and B;\nround 2, disputed only", rotation=90, ha="center", va="center",
        fontsize=7.5, style="italic")
# evaluator verdict after round 2
arrow(4.1, 3.7, 6.0, 3.7, text="still disputed\nafter round 2", tx=5.05, ty=2.95)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT)
