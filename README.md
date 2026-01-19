# energy-forecast-benchmark

Repository for the paper submission "Benchmarking Transformer and xLSTM for Time-Series Forecasting of Heat Consumption"

## Overview

This repository contains the code and experiments for benchmarking deep learning models for time-series forecasting of heating consumption. The study compares transformer-based architectures including Temporal Fusion Transformer (TFT), and Transformer, as well as the xLSTM on real-world building energy consumption data.

**Note:** The paper focuses specifically on hourly forecasting experiments. For a more comprehensive analysis including additional experiments, preprocessing details, and extended results, please refer to the full master thesis document (`master-thesis-wahl.pdf`).

## Repository Structure

```
energy-forecast-wahl/
├── src/energy_forecast/       # Source code
│   ├── model/                 # Model implementations and training
│   ├── data_processing/       # Data preprocessing utilities
│   ├── features.py           # Feature engineering
│   ├── config.py             # Configuration management
│   └── utils/                # Utility functions
├── notebooks/                 # Jupyter notebooks for analysis
└── references/               # Model configuration files (JSON/JSONL)
```

## Source Code

The main source code is located in [`src/energy_forecast/`](src/energy_forecast/):

### Core Training and Evaluation

- **[`model/train.py`](src/energy_forecast/model/train.py)** - Main training script for all models. This is the primary entry point for running experiments.
- **[`model/predict.py`](src/energy_forecast/model/predict.py)** - Prediction and inference script
- **[`model/evaluate.py`](src/energy_forecast/model/evaluate.py)** - Model evaluation utilities

### Model Implementations

- **[`model/models.py`](src/energy_forecast/model/models.py)** - Model architecture definitions
- **[`model/transformer.py`](src/energy_forecast/model/transformer.py)** - Transformer model
- **[`model/xlstm.py`](src/energy_forecast/model/xlstm.py)** - xLSTM model adapter: uses official xLSTM repository (https://github.com/NX-AI/xlstm)

### Data Processing

- **[`features.py`](src/energy_forecast/features.py)** - Feature engineering pipeline
- **[`dataset.py`](src/energy_forecast/dataset.py)** - Dataset loading and preparation
- **[`data_processing/`](src/energy_forecast/data_processing/)** - Data preprocessing utilities

### Configuration

- **[`config.py`](src/energy_forecast/config.py)** - Configuration management
- **[`references/`](references/)** - Model configuration files in JSON/JSONL format

## Notebooks

The [`notebooks/`](notebooks/) directory contains Jupyter notebooks for data exploration, analysis, and reproducibility:

### Data Exploration and Analysis

- **[`1.3-mw-data-exploration.ipynb`](notebooks/1.3-mw-data-exploration.ipynb)** - Comprehensive data exploration including preprocessing steps (outlier removal thresholds, interpolation rules) and configuration details

### Train/Test Split Analysis

- **[`1.0-mw-train-test-split-analysis.ipynb`](notebooks/1.0-mw-train-test-split-analysis.ipynb)** - Analysis of the train/validation/test split including:
  - Exact date ranges for train/val/test sets
  - Per-series counts and statistics
  - Number of samples after applying sliding windows

### Feature Engineering

- **[`1.0-mw-feature-analysis.ipynb`](notebooks/1.0-mw-feature-analysis.ipynb)** - Feature engineering analysis including the 10 Meteostat weather features used in the experiments

## Citation

If you use this code in your research, please cite:

```
@article{wahl2024benchmarking,
  title={Benchmarking Transformer and xLSTM for Time-Series Forecasting of Heat Consumption},
  author={Wahl, Marja and Bayer, Daniel and Rausch, Sven and Pruckner, Marco},
  year={2024}
}
```

## License

[Add license information here]
