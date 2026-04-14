# FLARE: A Statistical-AI Framework for Detecting Transient Flares in SDSS Stripe 82 Quasar Light Curves

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19494414.svg)](https://doi.org/10.5281/zenodo.19494414)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08196-b31b1b.svg)](https://arxiv.org/abs/2604.08196)


**Author:** Atal Agrawal  
**Affiliation:** Department of Physics, Indian Institute of Technology Roorkee

This repository contains the code and data for the FLARE (**F**lare detection via physics-informed **L**earning, **A**nomaly scoring, and **R**ecognition **E**ngine) framework, a three-stage pipeline for detecting transient flares in quasar light curves.

---

## Dataset

The dataset used for training, fine-tuning, and benchmarking is publicly available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.19494414

The dataset consists of fully simulated quasar light curves generated using a Damped Random Walk (DRW) process with synthetic flare and artifact injection (FRED, Gaussian, Gamma, and spike classes). It includes rendered light curve images and JSONL files containing prompts and labels for Vision Language Model (VLM) fine-tuning and evaluation.

---

## Overview

Quasars exhibit stochastic variability well-described by a Damped Random Walk (DRW). Occasionally, extreme luminosity changes — flares — represent significant departures from this baseline, offering insights into accretion disc dynamics and supermassive black hole physics. FLARE provides a modular, generalized framework to systematically detect these events.

Applied to the SDSS Stripe 82 dataset (9,258 spectroscopically confirmed quasars), the pipeline identifies **51 quasars** exhibiting distinct flaring activity using two complementary baselines.

## Pipeline

```
SDSS Stripe 82 Light Curves
        |
        v
  [1] Baseline Modeling
      ├── Physics-informed probabilistic GRU
      │   (trained on simulated DRW light curves)
      └── Iterative OU process
          (fitted directly to observed data with outlier masking)
        |
        v
  [2] Anomaly Scoring
      Standardized residuals → Extreme Value Theory
      Peaks-over-threshold GPD fit
      ├── GRU baseline: 8.69σ threshold → 51 candidates
      └── OU baseline:  3.73σ threshold → 92 candidates
        |
        v
  [3] Recognition Engine
      Dual VLM classifiers + Evaluator VLM
      2 iterative feedback rounds
        |
        v
    Confirmed Flares (51 unique)
```

### Stage 1: Baseline Modeling

Two complementary baselines model the DRW variability:

- **Physics-informed probabilistic GRU:** Trained on simulated DRW light curves, it predicts the next observation's mean and uncertainty using a composite loss combining negative log-likelihood, OU drift regularization, and variance regularization. Regularization weights are linearly annealed during training.
- **Iterative OU process:** Fitted directly to the observed r-band light curve using the celerite Gaussian process library, iteratively masking outliers to obtain robust DRW parameters even in the presence of flares.

### Stage 2: Anomaly Scoring

Standardized residuals from both baselines are analyzed using Extreme Value Theory (EVT) with a Peaks-over-Threshold approach. A Generalized Pareto Distribution (GPD) is fitted to tail exceedances, yielding detection thresholds of **8.69σ** (GRU) and **3.73σ** (iterative OU) at 1% false alarm probability. This flags 51 and 92 candidate flares respectively.

### Stage 3: Recognition Engine

Two VLMs independently classify each candidate light curve image as flare or non-flare. A third VLM evaluates the classifications, flags misclassifications, and provides feedback to the classifiers. This loop runs for **2 iterations**.

| Role | Model | Selection Rationale |
|------|-------|---------------------|
| Classifier A (High Recall) | Grok 4.1 Fast | Highest binary recall (~70%) |
| Classifier B (High Precision) | Qwen 3.5 Plus | Highest binary precision (~88%) |
| Evaluator | GPT-5 | Highest overall accuracy (42.8%) |

12 VLMs were benchmarked on a five-class classification task (non-flare, spike, Gaussian, Gamma, FRED) across 4,630 test light curves.

## Data

- **Source:** SDSS Stripe 82 quasar light curves from [MacLeod et al. (2010)](https://doi.org/10.1088/0004-637X/721/2/1014)
- **Band:** r-band photometry, with g-band cross-checks to rule out instrumental artifacts
- **Preprocessing:** Removal of bad observations (−99.99 / 99.99), Galactic extinction correction, MAD-based single-point spike removal
- **Simulations:** DRW light curves generated with [eztao](https://github.com/ywx649999311/EzTao) using per-object (τ, σ̂) parameters; synthetic FRED, Gaussian, and Gamma flares injected in flux space

## Repository Structure

```
│── Preprocessing.ipynb              # Data cleaning and preprocessing
│── Simulations_drw.ipynb            # DRW light curve simulations
│── Simulating_Flares&Spikes.ipynb   # Synthetic flare and spike injection
│── GRU_training.ipynb               # Physics-informed probabilistic GRU training
│── Iterative_drw.ipynb              # Iterative OU process baseline
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
- celerite (for iterative OU process)
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

- **Two complementary baselines**: a physics-informed probabilistic GRU (51 candidates at >8.69σ) and an iterative OU process (92 candidates at >3.73σ)
- **Recognition engine** classifies 22 flares from the GRU candidates (100% precision, 75.9% recall) and 29 from the OU candidates (55.2% precision, 59.3% recall)
- **51 confirmed flares** after human verification and g-band cross-checking, with only 5 detected by both baselines
- Flare rate: ~0.55% of the Stripe 82 sample

## Citation

If you use this code or the FLARE framework, please cite:

```
Agrawal, A. (2026). A Statistical-AI Framework for Detecting Transient Flares
in SDSS Stripe 82 Quasar Light Curves.
```

## License

This project is for academic research purposes.
