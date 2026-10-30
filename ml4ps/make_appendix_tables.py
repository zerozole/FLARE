"""Print the LaTeX rows for the camera-ready Appendix D tables (results on the OU 92, 27 flares).
CNN/ViT rows come from runs_cnn/<run>/result.json; VLM rows from runs/<run>/labels.csv scored as in score.py.
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import csv
import json
import math
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


def pct(x):
    return "---" if x != x else f"{100 * x:.0f}"


def cells(tp, fp, fn):
    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rl, rh = wilson(tp, tp + fn)
    pl, ph = wilson(tp, tp + fp)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    p_txt = "---" if tp + fp == 0 else f"{pct(prec)} [{pct(pl)}, {pct(ph)}]"
    return f"{tp} / {fp} & {pct(rec)} [{pct(rl)}, {pct(rh)}] & {p_txt} & {f1:.2f}"


print("% CNN / ViT")
CNN = [("ResNet-18, scratch", "scratch"), ("ResNet-18, ImageNet", "imagenet"), ("ViT-B/16, ImageNet", "vit")]
NAMES = {("scratch", "5class", 0): "scratch_5class_r4", ("scratch", "binary", 0): "scratch_binary_r2",
         ("scratch", "5class", 1): "scratch_axes_5class", ("scratch", "binary", 1): "scratch_axes_binary",
         ("imagenet", "5class", 0): "imagenet_5class", ("imagenet", "binary", 0): "imagenet_binary",
         ("imagenet", "5class", 1): "imagenet_axes_5class", ("imagenet", "binary", 1): "imagenet_axes_binary",
         ("vit", "5class", 0): "vit_5class_r2", ("vit", "binary", 0): "vit_binary_r2",
         ("vit", "5class", 1): "vit_axes_5class", ("vit", "binary", 1): "vit_axes_binary"}
for label, arch in CNN:
    for task in ("5class", "binary"):
        for axes in (0, 1):
            d = json.load(open(HERE / "runs_cnn" / NAMES[(arch, task, axes)] / "result.json"))
            r = d["real92"]
            print(f"{label} & {'five-class' if task == '5class' else 'binary'} & {'yes' if axes else 'no'} & "
                  f"{100 * d['test_acc']:.1f} & {cells(r['TP'], r['FP'], r['FN'])} & {r['roc_auc']:.2f} \\\\")

print("% VLM")


def score(run, col):
    rows = list(csv.DictReader(open(HERE / "runs" / run / "labels.csv", encoding="utf-8")))
    tp = sum(r[col] == "flare" and int(r["dbID"]) in FLARES for r in rows)
    fp = sum(r[col] == "flare" and int(r["dbID"]) not in FLARES for r in rows)
    fn = sum(r[col] != "flare" and int(r["dbID"]) in FLARES for r in rows)
    return cells(tp, fp, fn)


print(f"Engine, as published (Grok-4.1-fast) & yes & full & {cells(16, 13, 11)} \\\\")
VLM = [("Engine, rebuilt (GPT-5 mini)", "", "final_r2"), ("Gemini 4 Argon", "argon_", "single"),
       ("GPT-6 Astra", "astra_", "single"), ("Claude Fable 5.1", "fable_", "single")]
ENGINE_RUNS = {"axes_full": "axes_full_1", "noaxes_full": "noaxes_full_1", "axes_minimal": "axes_minimal_1",
               "noaxes_minimal": "noaxes_minimal_1"}
for label, prefix, col in VLM:
    for cond in ("noaxes_full", "axes_full", "noaxes_minimal", "axes_minimal"):
        run = ENGINE_RUNS[cond] if not prefix else prefix + cond
        axes = "no" if cond.startswith("noaxes") else "yes"
        prompt = cond.split("_")[1]
        print(f"{label} & {axes} & {prompt} & {score(run, col)} \\\\")
        if not prefix and cond == "axes_full":
            print(f"{label}, repeat & {axes} & {prompt} & {score('axes_full_2', col)} \\\\")
