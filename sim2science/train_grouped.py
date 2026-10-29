"""Retrain the CNN baselines on the quasar-grouped corpus (DATA_ROOT/grouped512) and score them
on the 92 real candidates.

  --arch imagenet : torchvision ResNet-18, IMAGENET1K_V1, ImageNet normalisation, lr 3e-4
                    (= train_resnet18_imagenet.py)
  --arch scratch  : the paper's from-scratch ResNet-18 (ReLU after the residual add, as trained
                    in train_cnnvit.py), mean 0.9 / std 0.1, lr 1e-3
  --task 5class   : non_flare / spike / gaussian / gamma / fred  (paper setting)
  --task binary   : flare (gaussian, gamma, fred) vs non-flare (non_flare, spike)  (reviewer Day3)

No horizontal flip for either arch (a flip reverses time). AdamW wd 0.01, cosine, batch 16, fp16.
Best-val checkpoint; early stop after --patience epochs without improvement.
"""
from paths import DATA_ROOT, WORK_ROOT  # noqa: E402
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

from cnn_models import ResNet18 as ScratchResNet18

DATA = (DATA_ROOT / "grouped512")
CAND = (DATA_ROOT / "plots_candidates_clean_512")
FLARES = {p.stem for p in (DATA_ROOT / "flare_plots").glob("*.pdf")}
CLASSES5 = ["non_flare", "spike", "gaussian", "gamma", "fred"]
FLARE_SET = {"gaussian", "gamma", "fred"}
DEV = torch.device("cuda")


def gpu_temp():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def cool_down(max_temp, waited):
    """Pause training while the GPU is above max_temp, until it is 3 C below it.
    The CPU shares the laptop's heat pipes, so this keeps both down."""
    t = gpu_temp()
    if t is None or t <= max_temp:
        return waited
    t0 = time.time()
    while t is not None and t > max_temp - 3:
        time.sleep(5)
        t = gpu_temp()
    return waited + time.time() - t0


def target(label, task):
    return CLASSES5.index(label) if task == "5class" else int(label in FLARE_SET)


class DS(Dataset):
    def __init__(self, split, task, tf):
        with open(DATA / split / "labels.csv", encoding="utf-8") as f:
            self.rows = [(DATA / split / "images" / r["file"], target(r["label"], task)) for r in csv.DictReader(f)]
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, y = self.rows[i]
        return self.tf(Image.open(p).convert("RGB")), y


def flare_prob(logits, task):
    p = torch.softmax(logits.float(), 1)
    return p[:, 2:].sum(1) if task == "5class" else p[:, 1]


@torch.no_grad()
def evaluate(model, loader, task, n_cls):
    model.eval()
    conf = np.zeros((n_cls, n_cls), int)
    for x, y in loader:
        with torch.amp.autocast("cuda"):
            pred = model(x.to(DEV)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            conf[t, p] += 1
    return conf.trace() / conf.sum(), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["imagenet", "scratch", "vit"], required=True)
    ap.add_argument("--task", choices=["5class", "binary"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gpu-temp", type=int, default=0, help="pause above this GPU temperature (0 = off)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, help="override the arch's default learning rate")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear warmup per step, then per-step cosine (0 = per-epoch cosine as before)")
    ap.add_argument("--axes", action="store_true",
                    help="train on grouped512_axes and score plots_candidates_axes_512 (the engine's renders)")
    args = ap.parse_args()
    global DATA, CAND
    if args.axes:
        DATA, CAND = (DATA_ROOT / "grouped512_axes"), (DATA_ROOT / "plots_candidates_axes_512")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)

    imagenet_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if args.arch == "imagenet":
        tf, lr = T.Compose([T.ToTensor(), imagenet_norm]), 3e-4
    elif args.arch == "vit":
        # ViT-B/16 is pretrained at 224 px; 512 -> 224 bilinear keeps the points ~3-4 px wide
        tf, lr = T.Compose([T.Resize((224, 224)), T.ToTensor(), imagenet_norm]), 1e-4
    else:
        tf, lr = T.Compose([T.ToTensor(), T.Normalize([0.9] * 3, [0.1] * 3)]), 1e-3
    if args.lr:
        lr = args.lr
    n_cls = 5 if args.task == "5class" else 2

    if args.arch == "imagenet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_cls)
    elif args.arch == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_cls)
    else:
        model = ScratchResNet18(n_cls)
    model.to(DEV)

    kw = dict(batch_size=16, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    tr = DataLoader(DS("train", args.task, tf), shuffle=True, **kw)
    va = DataLoader(DS("val", args.task, tf), shuffle=False, **kw)
    te = DataLoader(DS("test", args.task, tf), shuffle=False, **kw)
    print(f"{args.arch} {args.task}: train {len(tr.dataset)} val {len(va.dataset)} test {len(te.dataset)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if args.warmup_epochs:
        wu, total = args.warmup_epochs * len(tr), args.epochs * len(tr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / (total - wu))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    best, since, log = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ok = n = 0
        paused = 0.0
        for b, (x, y) in enumerate(tr):
            if args.max_gpu_temp and b % 25 == 0:
                paused = cool_down(args.max_gpu_temp, paused)
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                o = model(x)
                loss = crit(o, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if args.warmup_epochs:
                sched.step()
            ok += (o.argmax(1) == y).sum().item()
            n += len(y)
        if not args.warmup_epochs:
            sched.step()
        va_acc, _ = evaluate(model, va, args.task, n_cls)
        log.append({"epoch": ep, "train_acc": ok / n, "val_acc": va_acc, "minutes": (time.time() - t0) / 60,
                    "paused_minutes": paused / 60, "gpu_temp_end": gpu_temp()})
        print(f"  epoch {ep}: train {ok / n:.4f} val {va_acc:.4f} ({log[-1]['minutes']:.1f} min, "
              f"paused {paused / 60:.1f} min for heat, GPU {log[-1]['gpu_temp_end']} C)", flush=True)
        if va_acc > best:
            best, since = va_acc, 0
            torch.save(model.state_dict(), out / "best.pth")
        else:
            since += 1
            if since >= args.patience:
                print(f"  early stop: no val gain for {args.patience} epochs", flush=True)
                break

    model.load_state_dict(torch.load(out / "best.pth", map_location=DEV))
    te_acc, conf = evaluate(model, te, args.task, n_cls)

    # the 92 real candidates
    files = sorted(CAND.glob("*.png"))
    y = np.array([f.stem in FLARES for f in files], int)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(tf(Image.open(f).convert("RGB")).unsqueeze(0).to(DEV)) for f in files])
    pf = flare_prob(logits, args.task).cpu().numpy()
    am = logits.argmax(1).cpu().numpy()
    pred = (am >= 2) if args.task == "5class" else (am == 1)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    pos, neg = pf[y == 1], pf[y == 0]
    auc = float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
    order = np.argsort(-pf); ys = y[order]
    ap_ = float(np.sum((np.cumsum(ys) / np.arange(1, len(ys) + 1))[ys == 1]) / ys.sum())

    res = {"arch": args.arch, "task": args.task, "lr": lr, "warmup_epochs": args.warmup_epochs, "best_val": best, "test_acc": te_acc,
           "test_confusion": conf.tolist(), "epochs_run": len(log), "log": log,
           "real92": {"TP": tp, "FP": fp, "FN": int(y.sum()) - tp, "TN": int((y == 0).sum()) - fp,
                      "roc_auc": auc, "average_precision": ap_, "chance_ap": float(y.mean())}}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    with open(out / "real92_scores.csv", "w", encoding="utf-8") as f:
        f.write("id,true_flare,p_flare,argmax\n")
        for fi, yi, p, a in zip(files, y, pf, am):
            f.write(f"{fi.stem},{yi},{p:.5f},{a}\n")
    print(f"BEST VAL {best:.4f}  TEST {te_acc:.4f}", flush=True)
    print("test confusion (rows true):", conf.tolist(), flush=True)
    print(f"REAL 92: {res['real92']}", flush=True)


if __name__ == "__main__":
    main()
