"""Remaining trainings, one at a time, run as its own Windows process (independent of Claude Code).
Each job's output goes straight to runs_cnn/<name>.log; progress summary in runs_cnn/queue.log.
Jobs whose result.json already exists are skipped, so the queue can simply be started again.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
COOL = ["--workers", "4"]  # cooling governor removed at the user's request (2026-10-28 13:08)
JOBS = [
    ("scratch_5class_r4", ["--arch", "scratch", "--task", "5class"]),
    ("scratch_binary_r2", ["--arch", "scratch", "--task", "binary"]),
    ("vit_5class", ["--arch", "vit", "--task", "5class"]),
    ("vit_binary", ["--arch", "vit", "--task", "binary"]),
    ("RENDER", None),
    ("imagenet_axes_5class", ["--arch", "imagenet", "--task", "5class", "--axes"]),
    ("imagenet_axes_binary", ["--arch", "imagenet", "--task", "binary", "--axes"]),
]
runs = HERE / "runs_cnn"


def note(msg):
    with open(runs / "queue.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}\n")


for name, args in JOBS:
    if name == "RENDER":
        note("render axes start")
        with open(runs / "render.log", "a", encoding="utf-8") as log:
            subprocess.run([PY, "-u", "render_axes.py", "--procs", "6"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        note("render axes end")
        continue
    if (runs / name / "result.json").exists():
        note(f"{name} already done, skipped")
        continue
    if (runs / name).exists():
        note(f"{name} folder exists without a result; skipped, needs a new name")
        continue
    note(f"{name} start")
    with open(runs / f"{name}.log", "a", encoding="utf-8") as log:
        rc = subprocess.run([PY, "-u", "train_grouped.py", *args, "--out", str(runs / name), *COOL],
                            cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    note(f"{name} end, exit code {rc}")
note("all done")
