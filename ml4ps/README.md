# ML4PS 2026 camera-ready: From Random Walks to Reasoning

Code for "From Random Walks to Reasoning: Detecting Quasar Flares with Extreme Value Theory and
Vision-Language Models" (9th Workshop on Machine Learning and the Physical Sciences, ML4PS 2026).

The detection half of the paper (preprocessing, simulations, the iterative OU and physics-informed GRU
baselines, and the extreme-value thresholds of 3.73 sigma and 8.69 sigma) is in the notebooks in the
repository root. The scripts here add the verification experiments: CNNs, ViTs and VLMs on the 92 OU
candidates (27 human-labelled flares).

Set `FLARE_DATA_ROOT` (and, for the corpus build, `FLARE_WORK_ROOT`) before running; see `paths.py`.

## Trained models (Table 2, Table 6)

| Script | What it does |
|---|---|
| `build_grouped.py` | 80/10/10 split of the five-class corpus **by quasar** (37,013 / 4,648 / 4,619), seed 32 |
| `render_axes.py` | renders the corpus and the 92 candidates **with** time and magnitude axes, for the axis ablation |
| `cnn_models.py` | the from-scratch ResNet-18 |
| `train_grouped.py` | trains a ResNet-18 (scratch or ImageNet) or ViT-B/16 (ImageNet), five-class or binary, with or without axes, and scores the 92 candidates |
| `run_queue_*.py` | the training runs reported in the paper, one after another |

Example: `python train_grouped.py --arch vit --task 5class --lr 2e-5 --warmup-epochs 1 --axes --out runs_cnn/vit_axes_5class`

## VLMs (Tables 3, 4 and 5)

| Script | What it does |
|---|---|
| `run_engine.py` | the two-round recognition engine, or one model alone with `--single`; full prompt (read from `VLM_Recognition_Engine.ipynb`) or `--prompt minimal`; caches every API answer and stops at `--budget` USD |
| `score.py` | TP / FP / FN / TN, precision and recall with Wilson intervals, F1 |
| `bench_metrics.py` | five-class accuracy, macro-F1 and binary precision / recall on the synthetic benchmark |
| `make_appendix_tables.py` | the rows of Tables 2 and 3 |

The API key is read from `OPENROUTER_API_KEY` (environment or a local `.env`, never committed).
Example: `python run_engine.py --plots $FLARE_DATA_ROOT/plots_candidates --prompt full --single openai/gpt-6-astra --out runs/astra_axes_full --budget 5`

## Figures

`figures/make_engine_fig.py` and `make_pr_fig.py` (Figure 1), `make_qwen_conf.py` (QLoRA confusion
matrices, Appendix D).
