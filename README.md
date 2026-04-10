# FLARE: A Statistical-AI Framework for Detecting Transient Flares in SDSS Stripe 82 Quasar Light Curves

**Author:** Atal Agrawal  
**Affiliation:** Department of Physics, Indian Institute of Technology Roorkee

This repository contains the code and data for the FLARE (**F**lare detection via physics-informed **L**earning, **A**nomaly scoring, and **R**ecognition **E**ngine) framework, a three-stage pipeline for detecting transient flares in quasar light curves.

## Overview

Quasars exhibit stochastic variability well-described by a Damped Random Walk (DRW). Occasionally, extreme luminosity changes — flares — represent significant departures from this baseline, offering insights into accretion disc dynamics and supermassive black hole physics. FLARE provides a generalized framework to systematically detect these events.

Applied to the SDSS Stripe 82 dataset (9,258 spectroscopically confirmed quasars), the pipeline identifies **27 quasars** exhibiting distinct flaring activity.

## Pipeline

```
SDSS Stripe 82 Light Curves
        |
        v
  [1] Baseline Modeling
      Physics-informed probabilistic GRU
      with DRW/OU drift & variance regularization
        |
        v
  [2] Anomaly Scoring
      Standardized residuals → Extreme Value Theory
      Peaks-over-threshold GPD fit → 8.69σ threshold
        |
        v
  [3] Recognition Engine
      Dual VLM classifiers + Evaluator VLM
      5 iterative feedback rounds
        |
        v
    Confirmed Flares
```

### Stage 1: Baseline Modeling

A physics-informed probabilistic GRU models the DRW variability. The model predicts the next observation's mean and uncertainty, trained with a composite loss combining negative log-likelihood, OU drift regularization, and variance regularization. Regularization weights are linearly annealed during training.

### Stage 2: Anomaly Scoring

Standardized residuals from the GRU are analyzed using Extreme Value Theory (EVT) with a Peaks-over-Threshold approach. A Generalized Pareto Distribution (GPD) is fitted to tail exceedances, yielding a detection threshold of **8.69σ** at 1% false alarm probability. This flags 51 candidate flares from the Stripe 82 data.

### Stage 3: Recognition Engine

Two VLMs independently classify each candidate light curve image as flare or non-flare. A third VLM evaluates the classifications, flags misclassifications, and provides feedback to the classifiers. This loop runs for **5 iterations**.

| Role | Model | Selection Rationale |
|------|-------|---------------------|
| Classifier A (High Recall) | Grok 4.1 Fast | Highest binary recall (~70%) |
| Classifier B (High Precision) | Qwen 3.5 Plus | Highest binary precision (~88%) |
| Evaluator | GPT-5 | Highest overall accuracy (42.8%) |

12 VLMs were benchmarked on a five-class classification task (non-flare, spike, Gaussian, Gamma, FRED) across 4,630 test light curves.

## Data

- **Source:** SDSS Stripe 82 quasar light curves from [MacLeod et al. (2010)](https://doi.org/10.1088/0004-637X/721/2/1014)
- **Band:** r-band photometry, with g-band cross-checks to rule out instrumental artifacts
- **Preprocessing:** Removal of bad observations (-99.99), Galactic extinction correction, MAD-based single-point spike removal
- **Simulations:** DRW light curves generated with [eztao](https://github.com/ywx649999311/EzTao) using per-object (τ, σ) parameters; synthetic FRED, Gaussian, and Gamma flares injected in flux space

## Repository Structure

```

│── Preprocessing.ipynb              # Data cleaning and preprocessing
│── Simulations_drw.ipynb            # DRW light curve simulations
│── Simulating_Flares&Spikes.ipynb   # Synthetic flare and spike injection
│── GRU_training.ipynb               # Physics-informed probabilistic GRU training
│── Z_scores_qq_plot.ipynb           # Residual calibration and Q-Q plots
│── EVT_fitting.ipynb                # Extreme Value Theory threshold derivation
│── Qwen_qlora.ipynb                 # QLoRA fine-tuning of Qwen2.5VL-7b
│── Running_Inference.ipynb          # VLM benchmarking inference
│── VLM_Recognition_Engine.ipynb     # Multi-VLM recognition engine with feedback loop

```

## Requirements

- Python 3.12+
- PyTorch
- Transformers, PEFT, TRL (for QLoRA fine-tuning)
- OpenAI Python SDK (for OpenRouter API access)
- scipy, pandas, numpy, matplotlib, seaborn, scikit-learn
- python-dotenv, tqdm, eztao

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY = <your-openai-key>
   OPENROUTER_API_KEY = <your-openrouter-key>
   ```

## Key Results

- **51 candidates** flagged by the EVT anomaly scoring stage (>8.69σ)
- **30 candidates** classified as flares by the recognition engine
- **27 confirmed flares** after g-band cross-validation (3 rejected as instrumental artifacts)
- Flare rate: ~0.3% of the Stripe 82 sample

## Citation

If you use this code or the FLARE framework, please cite:

```
Agrawal, A. (2026). A Statistical-AI Framework for Detecting Transient Flares
in SDSS Stripe 82 Quasar Light Curves.
```

## License

This project is for academic research purposes.
