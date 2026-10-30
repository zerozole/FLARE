"""Rebuild the five-class synthetic corpus with a split BY QUASAR (reviewer Day3).

Same source folders, labels and class balancing as Preparing_dataset.py, but every image of a
given quasar goes to the same split, so no test quasar (its cadence, errors or DRW draw) is seen
in training. Images are resized 1024 -> 512 bilinear, as resize.py did for the original corpus.

Output: DATA_ROOT/grouped512\\{train,val,test}\\images\\*.png and labels.csv (file,label,quasar,folder).
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import csv
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

SRC = WORK_ROOT
OUT = (DATA_ROOT / "grouped512")
SEED = 32
FOLDERS = {
    "plots_set3": "non_flare",
    "plots_set4": "non_flare",
    "plots_set5": "non_flare",
    "plots_set6_single_point_spikes": "spike",
    "plots_set3_FRED_flares": "fred",
    "plots_set4_Gaussian_flares": "gaussian",
    "plots_set5_Gamma_flares": "gamma",
}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    rng = random.Random(SEED)

    by_label = defaultdict(list)
    for folder, label in FOLDERS.items():
        for p in sorted((SRC / folder).glob("*.png")):
            by_label[label].append((p, label, p.stem, folder))
    n_min = min(len(v) for v in by_label.values())
    samples = []
    for label in sorted(by_label):
        v = by_label[label]
        rng.shuffle(v)
        samples += v[:n_min]
    print(f"balanced to {n_min} per class, {len(samples)} images")

    quasars = sorted({s[2] for s in samples})
    rng.shuffle(quasars)
    n = len(quasars)
    split_of = {q: "train" for q in quasars[: int(0.8 * n)]}
    split_of.update({q: "val" for q in quasars[int(0.8 * n): int(0.9 * n)]})
    split_of.update({q: "test" for q in quasars[int(0.9 * n):]})

    rows = defaultdict(list)
    for s in samples:
        rows[split_of[s[2]]].append(s)
    tasks = []
    for split, ss in rows.items():
        (OUT / split / "images").mkdir(parents=True)
        rng.shuffle(ss)
        with open(OUT / split / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "quasar", "folder"])
            for i, (p, label, q, folder) in enumerate(ss):
                name = f"{i:06d}.png"
                w.writerow([name, label, q, folder])
                tasks.append((p, OUT / split / "images" / name))
        print(split, len(ss), dict(sorted(Counter(s[1] for s in ss).items())))

    assert not ({s[2] for s in rows["train"]} & {s[2] for s in rows["test"]})
    assert not ({s[2] for s in rows["train"]} & {s[2] for s in rows["val"]})

    def resize(t):
        src, dst = t
        with Image.open(src) as im:
            im.resize((512, 512), Image.Resampling.BILINEAR).save(dst)

    with ThreadPoolExecutor(8) as ex:
        for k, _ in enumerate(ex.map(resize, tasks), 1):
            if k % 5000 == 0:
                print(f"  resized {k}/{len(tasks)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
