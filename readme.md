This repository implements my Bil-476 term project. The project is split into two phases:

* **Part‑1 — Data Preprocessing & Feature Engineering**: builds a clean, feature‑rich dataset from raw CSVs.
* **Part‑2 — Deep Learning Models Training & Inference**: trains multiple PyTorch models with a fast pivoted data pipeline, evaluates them, and saves artifacts.

The pipeline reproduces the competition’s **16‑day forecast horizon** and evaluates performance with **RMSLE** including the **public (days 1‑5)** vs **private (days 6‑16)** split.

---

## Repository Layout

```
Data  Preprocessing & Feature Engineering (Part-1)/
  ├─ Data  Preprocessing & Feature Engineering Code.ipynb
  └─ Data  Preprocessing & Feature Engineering Code.py

Deep Learning Models Training & Inference (Part-2)/
  ├─ main.py               # Orchestrates runs: prepare_data, train, tune, evaluate
  ├─ data_utils.py         # Pivoting + label encoders + metadata I/O
  ├─ dataset.py            # Sliding-window dataset + DataLoaders (train/val/test)
  ├─ models.py             # 3 model architectures (GRU seq2seq, LSTM one-shot, Temporal CNN)
  ├─ train.py              # Trainer (RMSLE, early stopping, LR scheduling, checkpoints)
  └─ evaluate.py           # Evaluator (leaderboard, plots, error analysis)

Outputs (Evaluation Metrics & Plots)/   # Created after training/evaluation
saved_models/                           # Checkpoints, logs, plots per run
pivoted/                                # Pivoted .feather matrices + metadata (created by prepare_data)
```

---

## Data Requirements

Place the original Kaggle CSVs under a `Dataset/` directory at project root:

* `Dataset/train.csv`, `Dataset/test.csv`
* `Dataset/oil.csv`, `Dataset/holidays_events.csv`, `Dataset/stores.csv`, `Dataset/transactions.csv`
* (Optional) `Dataset/sample_submission.csv`

**Part‑1** produces `train_final.csv` and `test_final.csv`. **Part‑2** consumes these files and then creates a **pivoted** cache (wide matrices per feature) for fast training.

---

## Environment

* Python 3.9+
* PyTorch, NumPy, Pandas, scikit‑learn, tqdm, Matplotlib, Seaborn

Tip: create a virtual environment and `pip install` the usual DS stack.

---

## End‑to‑End Quickstart

1. **Run preprocessing (Part‑1)**

   * Notebook: open *Data  Preprocessing & Feature Engineering Code.ipynb* and run all cells, **or**
   * Script: `python "Data  Preprocessing & Feature Engineering (Part-1)/Data  Preprocessing & Feature Engineering Code.py"`
     Outputs: `train_final.csv`, `test_final.csv` (in project root or as configured in the script).

2. **Prepare pivoted training cache (Part‑2)**

   ```bash
   python "Deep Learning Models Training & Inference (Part-2)/main.py" \
     --prepare_data \
     --data_dir .
   ```

   This creates `pivoted/` (feature matrices in Feather format), `label_encoders.pkl`, and `metadata.json`.

3. **Train a model** (examples)

   ```bash
   # Temporal CNN (one-shot 16‑day prediction)
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py \
     --train --model_type cnn --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 64 --dropout 0.25

   # GRU Encoder–Decoder (autoregressive)
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py \
     --train --model_type seq2seq --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 128 --dropout 0.3

   # LSTM One‑Shot (parallel multi‑step)
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py \
     --train --model_type multi-step-seq2seq --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 128 --dropout 0.3
   ```

   Artifacts go to `saved_models/<run‑stamp>/` with `checkpoints/`, `logs/`, and `plots/`.

4. **Evaluate models / leaderboard / plots**

   ```bash
   # Evaluate all saved runs
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py --evaluate_all

   # Evaluate one model and produce plots + error analysis
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py \
     --evaluate_model saved_models/<model_dir> --plot_predictions --analyze_errors
   ```

5. **(Optional) Hyperparameter tuning**

   ```bash
   python Deep\ Learning\ Models\ Training\ \&\ Inference\ \(Part-2\)/main.py \
     --tune --tune_epochs 20 --tune_patience 6
   ```

---

## Design Highlights

* **Pivoted wide matrices** for each feature → \~orders‑of‑magnitude faster sliding windows.
* **Sliding‑window dataset** with random offsets (augmentation) and strict temporal splits.
* **Three architectures**:

  * *TemporalCnn* (one‑shot 16‑day)
  * *AutoRegressiveEncoderDecoderGruBased* (seq2seq, autoregressive)
  * *OneShotLstmBased* (parallel multi‑step)
* **Trainer** tracks RMSLE/R²/MAE/RMSE, plots history, saves best checkpoints.
* **Evaluator** produces a model leaderboard, prediction plots, and error analyses (incl. public vs private split).

See sub‑READMEs for per‑part details.

---

# README — Part‑1 (Data Preprocessing & Feature Engineering)

This module builds the **feature‑engineered training data** used by Part‑2.

## Inputs (expected in `Dataset/`)

* `train.csv`, `test.csv`
* `oil.csv`, `holidays_events.csv`, `stores.csv`, `transactions.csv`
* `sample_submission.csv` (optional)

## Outputs

* `train_final.csv` — enriched training data
* `test_final.csv` — enriched test data

## What the pipeline does

1. **Load and normalize dates** from all CSVs.
2. **Oil price imputation** using hybrid forward/backward fill + interpolation.
3. **Join** holiday/events, store metadata, transactions into the sales frame.
4. **Promotional dynamics**: rolling sums (`promo_roll_sum_7`, `promo_roll_sum_30`) over `onpromotion` by store–family.
5. **Time features**: cyclical encodings for month/day‑of‑week (sine & cosine).
6. **Interaction features**: e.g., `store_family_interaction`, and promo×state, promo×sum7 hybrids to capture cross‑effects.
7. **(EDA)** rich plots for oil, transactions, seasonalities (optional but included in the notebook).

> The exact final set of columns is the superset required by Part‑2’s pivoting and dataset code (see that README).

## How to run

### Option A — Jupyter

Open **`Data  Preprocessing & Feature Engineering Code.ipynb`** and **Run All**.

### Option B — Python script

```bash
python "Data  Preprocessing & Feature Engineering (Part-1)/Data  Preprocessing & Feature Engineering Code.py"
```

This produces `train_final.csv` and `test_final.csv` in your working directory (or the path configured in the notebook/script).

## Notes

* Ensure `date` columns remain `datetime64[ns]`.
* The script includes EDA for validation/insight; it’s safe to comment plotting out for headless runs.
* The outputs are **consumed by Part‑2** to build the pivoted training cache (Feather matrices + metadata).

---

# README — Part‑2 (Deep Learning Models Training & Inference)

This module trains and evaluates multiple neural models over a **pivoted** (wide) timeseries cache for speed and simplicity.

## Core Components

* **`main.py`** — the driver: *prepare\_data* → *train* → *evaluate* → *tune*.
* **`data_utils.py`** — `DataPivotProcessor`:

  * Fits **LabelEncoders** for categorical columns (e.g., store, family, city, state, store\_type, cluster; plus interaction keys).
  * **Pivots** each time‑varying feature into a wide matrix (rows = store–family, columns = dates).
  * Saves **`pivoted/*.feather`**, **`label_encoders.pkl`**, and **`metadata.json`**.
* **`dataset.py`** — `SalesDataset`:

  * Creates **sliding windows** of `timesteps` history (default 200) → predict **16 days** ahead.
  * Uses **random time offsets** during training (augmentation).
  * Builds **train/val/test** with non‑overlapping prediction start dates (no leakage).
* **`models.py`** — three architectures:

  * **TemporalCnn** (one‑shot 16‑day prediction) using dilated 1D convolutions + static embeddings.
  * **AutoRegressiveEncoderDecoderGruBased** (seq2seq, autoregressive).
  * **OneShotLstmBased** (parallel multi‑step with LSTM backbone).
  * All models embed static categorical features (city, state, store\_type, cluster, store\_nbr, family) and may include interaction embeddings.
* **`train.py`** — `Trainer` with **RMSLE** loss, early stopping, ReduceLROnPlateau, gradient clipping, plots, and checkpointing.
* **`evaluate.py`** — `ModelEvaluator` to score **public (1‑5)** vs **private (6‑16)** days, build a **leaderboard**, and generate **plots** including error analysis.

## Data Prep (must run once per dataset change)

```bash
python main.py --prepare_data --data_dir .
# creates: pivoted/*.feather, label_encoders.pkl, metadata.json
```

## Training Examples

```bash
# Temporal CNN (one-shot)
python main.py --train --model_type cnn --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 64 --dropout 0.25

# GRU encoder–decoder (autoregressive)
python main.py --train --model_type seq2seq --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 128 --dropout 0.3

# LSTM one-shot (parallel multi-step)
python main.py --train --model_type multi-step-seq2seq --epochs 30 --batch_size 32 --lr 0.001 --latent_dim 128 --dropout 0.3
```

Artifacts for each run are stored under `saved_models/<model>_lr{...}_dim{...}_drop{...}_bs{...}_{timestamp}/` with:

* `checkpoints/best_model.pth` + metrics JSON
* `plots/*` (training curves, per‑day RMSLE, sample predictions)
* `logs/*`

## Evaluation & Analysis

```bash
# Evaluate everything under saved_models/
python main.py --evaluate_all

# Evaluate one model and generate visualizations
python main.py --evaluate_model saved_models/<model_dir> --plot_predictions --analyze_errors
```

Produces:

* `evaluation_results/model_leaderboard.csv`
* Per‑model plots: predictions vs actuals (with public/private split), error distributions, per‑day RMSLE.

## Key Implementation Choices

* **RMSLE** matches the competition metric and is computed per‑day, public vs private, and overall.
* **Static embeddings** for categorical store/product context + **temporal CNN/RNN** for sequence dynamics.
* **Sliding windows** over a **pivoted** cache to drastically speed up data access and batching.
* **Strict temporal splits** to avoid leakage and reflect the evaluation protocol.

You're ready to iterate on features, models, and tuning—drop new features in Part‑1, re‑`--prepare_data`, and retrain here.
