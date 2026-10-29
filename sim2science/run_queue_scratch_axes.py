"""From-scratch ResNet-18 on the axis renders, started after the ViT axis queue (run_queue_vit_axes.py) ends.
Same settings as scratch_5class_r4 / scratch_binary_r2 (arch default lr 1e-3, per-epoch cosine).
Own Windows process, same logging as the other queues.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
ARGS = ["--workers", "4", "--axes"]
JOBS = [
    ("scratch_axes_5class", ["--arch", "scratch", "--task", "5class"]),
    ("scratch_axes_binary", ["--arch", "scratch", "--task", "binary"]),
]
runs = HERE / "runs_cnn"


def note(msg):
    with open(runs / "queue.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


def vit_queue_done():
    lines = (runs / "queue.log").read_text(encoding="utf-8").splitlines()
    return any(line.endswith("vit axes runs done") for line in lines)


note("scratch axes queue waiting for the ViT axis runs")
while not vit_queue_done():
    time.sleep(60)

for name, args in JOBS:
    if (runs / name).exists():
        note(f"{name} folder exists; skipped, needs a new name")
        continue
    note(f"{name} start")
    with open(runs / f"{name}.log", "a", encoding="utf-8") as log:
        rc = subprocess.run([PY, "-u", "train_grouped.py", *args, "--out", str(runs / name), *ARGS],
                            cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    note(f"{name} end, exit code {rc}")
note("scratch axes runs done")
