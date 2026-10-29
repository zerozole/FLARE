"""ViT-B/16 reruns (the first two collapsed to one class at lr 1e-4, no warmup): lr 2e-5 with one
epoch of linear warmup. Own Windows process, same logging as run_queue_cool.py.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
ARGS = ["--workers", "4", "--lr", "2e-5", "--warmup-epochs", "1"]
JOBS = [
    ("vit_5class_r2", ["--arch", "vit", "--task", "5class"]),
    ("vit_binary_r2", ["--arch", "vit", "--task", "binary"]),
]
runs = HERE / "runs_cnn"


def note(msg):
    with open(runs / "queue.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


for name, args in JOBS:
    if (runs / name).exists():
        note(f"{name} folder exists; skipped, needs a new name")
        continue
    note(f"{name} start")
    with open(runs / f"{name}.log", "a", encoding="utf-8") as log:
        rc = subprocess.run([PY, "-u", "train_grouped.py", *args, "--out", str(runs / name), *ARGS],
                            cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    note(f"{name} end, exit code {rc}")
note("vit reruns done")
