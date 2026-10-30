"""Macro-F1 and per-class recall for every VLM in the synthetic benchmark (Table 4),
from the saved per-image predictions; no API calls.

Also checks that the accuracy we recompute matches the paper's Table 4.
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import csv
import json
from collections import Counter
from pathlib import Path

R = (WORK_ROOT / "Results")
CLASSES = ["non_flare", "spike", "gaussian", "gamma", "fred"]
PAPER = {"GPT-5": 42.8, "GPT-5-nano": 32.6, "GPT-5 mini": 32.6, "Kimi-k2.5": 31.4,
         "Qwen3-VL-30B-A3B": 29.5, "Qwen3-VL-235B-A22B": 28.0, "Qwen-3.5-plus": 27.9,
         "Mistral Large 3": 27.5, "Grok-4.1-fast": 25.2, "Claude 3 Haiku": 23.3,
         "Qwen2.5-VL-7b-Instruct": 19.8, "Qwen2.5-VL-7b-QLoRA": 18.4}


def from_json(path, pred_key="pred"):
    return [(r["image"].split("/")[-1], r["ground_truth"], r.get(pred_key) or "") for r in
            json.load(open(path, encoding="utf-8"))]


def from_csv(path):
    with open(path, encoding="utf-8") as f:
        return [(r["image_path"].split("/")[-1], r["ground_truth"], r["prediction"]) for r in csv.DictReader(f)]


SOURCES = {
    "GPT-5": lambda: from_json(R / "gpt5_predictions.json"),
    "GPT-5 mini": lambda: from_json(R / "json_files/gpt5mini_predictions.json"),
    "GPT-5-nano": lambda: from_csv(R / "gpt5nano_results/predictions.csv"),
    "Claude 3 Haiku": lambda: from_json(R / "json_files/haiku3_predictions.json"),
    "Grok-4.1-fast": lambda: from_json(R / "json_files/grok41fast_predictions.json"),
    "Qwen-3.5-plus": lambda: from_json(R / "json_files/qwen35plus_predictions.json"),
    "Qwen3-VL-235B-A22B": lambda: from_json(R / "qwen3vl235ba22_predictions.json"),
    "Qwen3-VL-30B-A3B": lambda: from_json(R / "json_files/qwen3vl30b_predictions.json"),
    "Mistral Large 3": lambda: from_json(R / "Mistral3Large_predictions.json"),
    "Kimi-k2.5": lambda: from_json(R / "json_files/kimi25_predictions.json"),
    "Qwen2.5-VL-7b-Instruct": lambda: from_json(R / "inference_lora_model_v2/base_results.json", "pred_class"),
    "Qwen2.5-VL-7b-QLoRA": lambda: from_json(R / "inference_lora_model_v2/lora_results.json", "pred_class"),
}


def metrics(rows):
    n = len(rows)
    acc = sum(t == p for _, t, p in rows) / n
    rec, f1 = {}, {}
    for c in CLASSES:
        tp = sum(t == c and p == c for _, t, p in rows)
        fn = sum(t == c and p != c for _, t, p in rows)
        fp = sum(t != c and p == c for _, t, p in rows)
        rec[c] = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    # binary: any flare class vs not
    flare = {"gaussian", "gamma", "fred"}
    btp = sum(t in flare and p in flare for _, t, p in rows)
    bfp = sum(t not in flare and p in flare for _, t, p in rows)
    bfn = sum(t in flare and p not in flare for _, t, p in rows)
    bprec = btp / (btp + bfp) if btp + bfp else 0.0
    brec = btp / (btp + bfn) if btp + bfn else 0.0
    return n, acc, sum(f1.values()) / 5, rec, Counter(p for _, _, p in rows), bprec, brec


out_rows = []
print(f"{'model':24s} {'n':>5s} {'acc':>6s} {'paper':>6s} {'macroF1':>8s}  "
      + " ".join(f"{c[:6]:>7s}" for c in CLASSES) + "   top predicted class   bin P / R")
for name, load in SOURCES.items():
    try:
        rows = load()
    except FileNotFoundError as e:
        print(f"{name:24s} missing: {e.filename}")
        continue
    n, acc, mf1, rec, pc, bp, br = metrics(rows)
    top, cnt = pc.most_common(1)[0]
    print(f"{name:24s} {n:5d} {acc:6.1%} {PAPER[name]:5.1f}% {mf1:8.3f}  "
          + " ".join(f"{rec[c]:7.1%}" for c in CLASSES)
          + f"   {top} {cnt / n:.0%}   {bp:.0%} / {br:.0%}")
    out_rows.append({"model": name, "n": n, "accuracy": round(acc, 4), "paper_accuracy": PAPER[name] / 100,
                     "macro_f1": round(mf1, 4), **{f"recall_{c}": round(rec[c], 4) for c in CLASSES},
                     "most_predicted": top, "most_predicted_frac": round(cnt / n, 4),
                     "binary_flare_precision": round(bp, 4), "binary_flare_recall": round(br, 4)})

with open(Path(__file__).with_name("bench_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
