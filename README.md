# Graph-Based Financial Forecasting

**TU Delft - DSAIT4305 Graph Machine Learning**
**Group 21**: Benghe Cristian, Ciuntu Sorin-Andrei, Darie Dan, Jonsson Arnar, Petriceanu Alexandru

Can learned graphs used in GNNs beat a naive CMA-ES optimizer in financial forecasting?

## Project Structure

```
project/
├── configs/
│   └── default.yaml          # All hyperparameters and ticker lists
├── src/
│   ├── data/
│   │   ├── download.py        # Fetch stock data from Yahoo Finance
│   │   ├── features.py        # Log returns, rolling volatility, normalized volume
│   │   └── dataset.py         # Sliding-window PyTorch dataset
│   ├── models/
│   │   ├── mtgnn.py           # MTGNN (Wu et al., 2020) with learned graph
│   │   └── lstm.py            # LSTM baseline (no graph)
│   ├── baselines/
│   │   ├── var.py             # Vector Autoregression baseline
│   │   └── cmaes_search.py    # CMA-ES derivative-free graph search
│   └── evaluation/
│       └── metrics.py         # MAE, RMSE, directional accuracy, Sharpe
├── scripts/
│   ├── download_data.py       # Step 1: download and preprocess data
│   └── train_mtgnn.py         # Step 2: train MTGNN
└── notebooks/                 # Exploration and visualization
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download and preprocess stock data
python scripts/download_data.py

# 2. Train MTGNN
python scripts/train_mtgnn.py
```

## Models

| Model | Type | Graph |
|-------|------|-------|
| **MTGNN** | GNN + temporal conv | Learned end-to-end |
| **LSTM** | Recurrent neural net | None |
| **VAR** | Linear multivariate | None |
| **CMA-ES + GNN** | Evolutionary search | Optimized by CMA-ES |

## Data

20 companies across two industries (10 semiconductor, 10 agriculture). Daily data from 2015-2026.

**Features per node**: log return, rolling volatility, normalized volume.

**Task**: Node-level regression. Given a 20-day window of all companies, predict next-day log returns.

## Evaluation

- **MAE** and **RMSE** for forecast accuracy
- **Directional accuracy** for up/down prediction
- **Sharpe ratio** for optional portfolio evaluation
