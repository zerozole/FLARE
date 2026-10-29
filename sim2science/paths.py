"""Where the data live. Set FLARE_DATA_ROOT (and optionally FLARE_WORK_ROOT) before running.

DATA_ROOT holds the rendered corpus and candidate plots: grouped512/, grouped512_axes/, plots_candidates/,
plots_candidates_clean_512/, plots_candidates_axes_512/ and flare_plots/ (one file per human-labelled flare).
WORK_ROOT is the original working directory of the notebooks (SIMULATIONS/, Results/, the training PNGs).
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FLARE_DATA_ROOT", "data"))
WORK_ROOT = Path(os.environ.get("FLARE_WORK_ROOT", str(DATA_ROOT / "Work_directory")))
