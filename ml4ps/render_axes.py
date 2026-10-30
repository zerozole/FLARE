"""Render the quasar-grouped corpus WITH axes, in the style of the engine's real-candidate plots
(plots_candidates: serif font, red points with error bars, ID as title, "r magnitude" vs MJD,
inverted magnitude axis, inward ticks, dashed grid), then resize to 512x512 bilinear exactly as
the real candidate plots are resized. Lets a CNN be trained on what the VLMs saw (reviewers'
axis confound, in the CNN direction).

Usage: python render_axes.py --sample 1000679      (one image per class to check the look)
       python render_axes.py                       (the whole corpus + the 92 real candidates)
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import argparse
import csv
import io
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SIM = (WORK_ROOT / "SIMULATIONS")
GROUPED = (DATA_ROOT / "grouped512")
OUT = (DATA_ROOT / "grouped512_axes")
REAL_IN = (DATA_ROOT / "plots_candidates")
REAL_OUT = (DATA_ROOT / "plots_candidates_axes_512")
# image folder -> (simulation folder, magnitude column)
SOURCE = {
    "plots_set3": ("simulated_DRW_set3", "mag_r_sim"),
    "plots_set4": ("simulated_DRW_set4", "mag_r_sim"),
    "plots_set5": ("simulated_DRW_set5", "mag_r_sim"),
    "plots_set6_single_point_spikes": ("simulated_DRW_set6_single_point_spikes", "mag_r_with_flare"),
    "plots_set3_FRED_flares": ("simulated_DRW_set3_FRED_flares", "mag_r_with_flare"),
    "plots_set4_Gaussian_flares": ("simulated_DRW_set4_Gaussian_flares", "mag_r_with_flare"),
    "plots_set5_Gamma_flares": ("simulated_DRW_set5_Gamma_flares", "mag_r_with_flare"),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.linewidth": 1.2,
})


def render(t, m, e, title):
    fig, ax = plt.subplots(figsize=(6.3, 3.5), dpi=300)
    ax.errorbar(t, m, yerr=e, fmt="o", color="#e8604c", ms=4, mec="#d9472f", mew=0.6,
                elinewidth=0.8, alpha=0.85, capsize=0)
    ax.invert_yaxis()
    ax.set_xlabel("MJD", fontsize=14)
    ax.set_ylabel(r"$r$ magnitude", fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.grid(True, ls="--", lw=0.5, color="#cccccc", alpha=0.7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_sim(folder, quasar):
    simdir, col = SOURCE[folder]
    with open(SIM / simdir / f"{quasar}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["MJD"]) for r in rows])
    m = np.array([float(r[col]) for r in rows])
    e = np.array([float(r["magr_err"]) for r in rows])
    return t - t.min(), m, e


def job(args):
    folder, quasar, dst = args
    if dst.exists():
        return
    t, m, e = load_sim(folder, quasar)
    render(t, m, e, quasar).resize((512, 512), Image.Resampling.BILINEAR).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    if args.sample:
        out = Path(__file__).with_name("axes_samples")
        out.mkdir(exist_ok=True)
        for folder in SOURCE:
            t, m, e = load_sim(folder, args.sample)
            render(t, m, e, args.sample).save(out / f"{folder}_{args.sample}.png")
        print(f"samples in {out}")
        return

    tasks = []
    for split in ("train", "val", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        with open(GROUPED / split / "labels.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(OUT / split / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        tasks += [(r["folder"], r["quasar"], OUT / split / "images" / r["file"]) for r in rows]

    REAL_OUT.mkdir(exist_ok=True)
    for p in sorted(REAL_IN.glob("*.png")):
        with Image.open(p) as im:
            im.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR).save(REAL_OUT / p.name)
    print(f"real candidates resized: {len(list(REAL_OUT.glob('*.png')))}", flush=True)

    with Pool(args.procs) as pool:
        for k, _ in enumerate(pool.imap_unordered(job, tasks, chunksize=64), 1):
            if k % 5000 == 0:
                print(f"  rendered {k}/{len(tasks)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
