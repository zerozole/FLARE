"""Score engine runs against the human labels.

Truth: a candidate is a flare if its ID has a PDF in ../flare_plots (27 of the 92).
For each run: classifier A alone, B alone, round 1 final, round 2 final (full engine).
Prints TP/FP/FN/TN, precision and recall with 95% Wilson intervals, F1, accuracy.

Usage: python score.py runs/axes_full_1 runs/noaxes_full_1 ...
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
FLARES = {int(p.stem) for p in (DATA_ROOT / "flare_plots").glob("*.pdf")}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(rows, col):
    tp = fp = fn = tn = unc = 0
    for r in rows:
        pred = r[col] == "flare"
        if r[col] not in ("flare", "non_flare"):
            unc += 1
        true = int(r["dbID"]) in FLARES
        tp += pred and true
        fp += pred and not true
        fn += (not pred) and true
        tn += (not pred) and not true
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    acc = (tp + tn) / len(rows)
    pl, ph = wilson(tp, tp + fp)
    rl, rh = wilson(tp, tp + fn)
    return (f"TP {tp:2d} FP {fp:2d} FN {fn:2d} TN {tn:2d} | "
            f"precision {prec:6.1%} [{pl:.0%}-{ph:.0%}]  recall {rec:6.1%} [{rl:.0%}-{rh:.0%}]  "
            f"F1 {f1:.3f}  acc {acc:.1%}" + (f"  (uncertain/error {unc})" if unc else ""))


for run in sys.argv[1:]:
    rows = list(csv.DictReader(open(Path(run) / "labels.csv", encoding="utf-8")))
    n_true = sum(int(r["dbID"]) in FLARES for r in rows)
    print(f"\n{run}  (n={len(rows)}, flares={n_true})")
    cols = ((("single model", "single"),) if "single" in rows[0] else
            (("A alone (recall, r1)", "a_r1"), ("B alone (precision, r1)", "b_r1"),
             ("engine, round 1", "final_r1"), ("engine, round 2", "final_r2")))
    for name, col in cols:
        print(f"  {name:20s} {score(rows, col)}")
    acc_all_non = sum(int(r["dbID"]) not in FLARES for r in rows) / len(rows)
    f1_accept = 2 * n_true / (len(rows) + n_true)
    print(f"  {'baselines':20s} accept-all F1 {f1_accept:.3f}; all-non-flare accuracy {acc_all_non:.1%}")
