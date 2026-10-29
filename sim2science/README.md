# Sim2Science 2026 camera-ready: CNNs, ViTs and VLMs on the real flare candidates

Code for "Beyond Pattern Matching: Bridging the Sim-to-Real Gap in Transient Flare Verification Using
Vision-Language Models" (NeurIPS 2026 Workshop on Sim2Science). The notebooks in the repository root
build the simulated corpus, the candidate list and the original recognition engine; the scripts here
add the controls and ablations of the camera-ready.

Set `FLARE_DATA_ROOT` (and, for the corpus build, `FLARE_WORK_ROOT`) before running; see `paths.py`.
All evaluation is on the 92 candidates of the iterative OU baseline (27 human-labelled flares).

## Trained models (Tables 2, 5 and 6)

| Script | What it does |
|---|---|
| `build_grouped.py` | 80/10/10 split of the five-class corpus **by quasar** (37,013 / 4,648 / 4,619), seed 32 |
| `render_axes.py` | renders the corpus and the 92 candidates **with** time and magnitude axes, for the axis ablation |
| `cnn_models.py` | the from-scratch ResNet-18 |
| `train_grouped.py` | trains a ResNet-18 (scratch or ImageNet) or ViT-B/16 (ImageNet), five-class or binary, with or without axes, and scores the 92 candidates |
| `run_queue_*.py` | the training runs reported in the paper, one after another |

Example: `python train_grouped.py --arch vit --task 5class --lr 2e-5 --warmup-epochs 1 --axes --out runs_cnn/vit_axes_5class`

## VLMs (Tables 2, 4 and 7)

| Script | What it does |
|---|---|
| `run_engine.py` | the two-round recognition engine, or one model alone with `--single`; full prompt (read from `VLM_Recognition_Engine.ipynb`) or `--prompt minimal`; caches every API answer and stops at `--budget` USD |
| `score.py` | TP / FP / FN / TN, precision and recall with Wilson intervals, F1 |
| `bench_metrics.py` | five-class accuracy, macro-F1 and binary precision / recall on the synthetic benchmark |
| `make_appendix_tables.py` | the rows of Tables 6 and 7 |

The API key is read from `OPENROUTER_API_KEY` (environment or a local `.env`, never committed).
Example: `python run_engine.py --plots $FLARE_DATA_ROOT/plots_candidates --prompt full --single openai/gpt-6-astra --out runs/astra_axes_full --budget 5`

## Figures

`figures/make_triple_fig.py` (Figure 1), `make_engine_fig.py` and `make_pr_fig.py` (Figure 2),
`make_qwen_conf.py` (Figure 3).
