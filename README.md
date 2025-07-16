# Benchmarking Transformers and xLSTM on Forecasting of Heat Load Data

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive machine learning project for energy consumption forecasting using advanced time series models including Temporal Fusion Transformers (TFT), xLSTM, and transformer architectures.

## Overview

This project implements and evaluates multiple deep learning models for energy consumption forecasting, with a focus on building-level energy prediction. The project leverages time series data with weather features, building characteristics, and temporal patterns to predict energy consumption at different horizons (hourly, daily).

### Key Features

- **Multiple Model Architectures**: Supports TFT, xLSTM, Transformers, LSTM, FCN, and traditional ML models
- **Flexible Data Processing**: Handles both hourly and daily resolution data with interpolation options
- **Feature Engineering**: Comprehensive weather, building, and temporal feature sets
- **Experiment Tracking**: Integration with Weights & Biases (wandb) for experiment management
- **Clustering Analysis**: Building clustering for targeted forecasting strategies
- **Missing Data Handling**: Robust preprocessing for real-world energy data with gaps

## Project Structure

```
├── README.md          <- This file
├── requirements.txt   <- Python dependencies
├── data
│   ├── external       <- External data sources (weather, holidays)
│   ├── processed      <- Processed datasets ready for modeling
│   └── legacy_data.zip <- Historical data archive
│
├── notebooks          <- Jupyter notebooks for analysis and exploration
│   ├── 1.0-mw-data-exploration.ipynb
│   ├── 1.0-mw-correlation-analysis.ipynb
│   ├── 1.0-mw-clustering.ipynb
│   └── ...
│
├── references         <- Configuration files and metadata
│   ├── configs.jsonl  <- Model configuration parameters
│   └── ...
│
├── reports            <- Generated analysis and results
│   ├── figures        <- Generated plots and visualizations
│   └── *.csv          <- Evaluation results and metrics
│
└── src/energy_forecast <- Source code
    ├── config.py              <- Configuration and constants
    ├── dataset.py             <- Data loading and preprocessing
    ├── features.py            <- Feature engineering
    ├── plots.py               <- Visualization utilities
    ├── data_processing/       <- Data processing modules
    ├── model/                 <- Model implementations
    │   ├── train.py           <- Training pipeline
    │   ├── models.py          <- Model definitions
    │   ├── evaluate.py        <- Evaluation utilities
    │   ├── transformer.py     <- Transformer implementation
    │   ├── xlstm.py          <- xLSTM implementation
    │   └── tft_xlstm_fusion.py <- TFT-xLSTM fusion model
    ├── tft/                   <- Temporal Fusion Transformer
    └── utils/                 <- Utility functions
```

## Installation

> This project requires [Git LFS](https://git-lfs.com/) to be installed for downloading of large data files

1. Clone the repository:
```bash
git clone git@gitlab2.informatik.uni-wuerzburg.de:s371469/energy-forecast-wahl.git
cd energy-forecast-wahl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up [xLSTM repository](https://github.com/NX-AI/xlstm)
```bash
cd src/energy_forecast
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
```

4. Download data with Git LFS #TODO

## Usage

### Data Preparation

The project expects processed data in the `data/processed/` directory. Key datasets include:
- `building_daily_7_7.csv` - Daily resolution building data
- `building_hourly_72_72.csv` - Hourly resolution building data

### Training Models

#### Single Model Training

Train a specific model with a configuration file from `references/your_config.json`:
```bash
python src/energy_forecast/model/train.py --config_file your_config
```

Example configuration file: `references/xlstm.json`

#### Batch Training

Train multiple models using configurations from `references/configs.jsonl`:
```bash
python src/energy_forecast/model/train.py
```

TODO: example content for file

### Configuration

Model configurations are stored in JSON format in the `references/` directory. Key parameters include:

- `model`: Model type (`transformer`, `xlstm`, `lstm`, `FCN3`)
- `res`: Input data format resolution (`daily`, `hourly`)
- `n_in`: Input sequence length
- `n_out`: Output sequence length (forecast horizon)
- `n_future`: Future variables length
- `feature_code`: Feature set identifier (see `config.py`)
- `epochs`: Training epochs
- `batch_size`: Batch size for training, defaults to

  **Optional**:
- `dataset`: Dataset type (default: `building`, `meta`, `missing_data_90`)
- `neurons`: Number of neurons (for `FCN3`, `lstm`), default: 100
- `dropout`: Dropout percentage for corrseponding dropout layers, default: 0.1
- `num_heads`: Number of heads for multi-head attention (`transformer`, `xlstm`), default: 4
- `log`: whether to log to WandB. Need to be logged-in for logging to work, default: False
- `project`: WandB project tag
- `remove_per`: Remove percentage for experiments with removed training data, default: 0.0

More detail about configuration values can be found in `src/energy_forecast/model/train.py` class `TrainConfig`

Example configuration:
```json
{
  "model": "transformer",
  "dataset": "building",
  "res": "daily",
  "n_in": 7,
  "n_out": 7,
  "n_future": 7, 
  "feature_code": 14,
  "epochs": 10,
  "batch_size": 64,
  "dropout": 0.1,
  "num_heads": 4
}
```

### Available Models

1. **Temporal Fusion Transformer (TFT)**: State-of-the-art attention-based model for interpretable forecasting
2. **xLSTM**: Extended LSTM with improved memory capabilities
3. **Transformer**: Standard transformer encoder architecture for time series
4. **LSTM/RNN**: Traditional recurrent neural networks
5. **FCN**: Fully connected network

> The **TFT** training is not included in the framework described above. It needs to be separately trained using the script `tft/script_train_fixed_params.py`. More instructions can be found in the `tft/README.md`

### Feature Sets

The project includes 16 predefined feature sets (see `config.py`):
- Weather features (temperature, humidity, precipitation, etc.)
- Building characteristics (area, type, energy source)
- Temporal features (weekday, month, holidays)
- Lag features and differencing

Important feature codes:

- `12`: only past energy consumption
- daily experiments: `13` (all features), `14` (all features but building features)
- hourly experiments: `15` (all features), `16` (all features but building features)

## Repository Structure

This section provides a detailed overview of the repository organization, including all folders and key files with their purposes.

### Root Directory Files

- [`.gitattributes`](.gitattributes) - Git configuration for file handling
- [`README.md`](README.md) - This documentation file
- [`requirements.txt`](requirements.txt) - Python package dependencies

### Data Directory ([`data/`](data))

Contains all project datasets and external data sources:

#### External Data ([`data/external/`](data/external))
- [`features/holidays.csv`](data/external/features/holidays.csv) - Holiday calendar data for feature engineering

#### Processed Data ([`data/processed/`](data/processed))
- [`building_daily_7_7.csv`](data/processed/building_daily_7_7.csv) - Daily resolution building energy data (7-day input/output)
- [`building_hourly_72_72.csv`](data/processed/building_hourly_72_72.csv) - Hourly resolution building energy data (72-hour input/output)

#### Legacy Data
- [`legacy_data.zip`](data/legacy_data.zip) - Archived historical datasets

### Notebooks Directory ([`notebooks/`](notebooks))

Jupyter notebooks for analysis and experimentation:

#### Data Exploration
- [`1.0-mw-data-exploration.ipynb`](notebooks/1.0-mw-data-exploration.ipynb) - Initial data exploration and visualization
- [`1.1-mw-data-exploration.ipynb`](notebooks/1.1-mw-data-exploration.ipynb) - Extended data exploration
- [`1.2-mw-data-exploration.ipynb`](notebooks/1.2-mw-data-exploration.ipynb) - Additional data analysis

#### Analysis Notebooks
- [`1.0-mw-correlation-analysis.ipynb`](notebooks/1.0-mw-correlation-analysis.ipynb) - Feature correlation analysis
- [`1.1-mw-correlation-analysis.ipynb`](notebooks/1.1-mw-correlation-analysis.ipynb) - Extended correlation analysis
- [`1.2-mw-correlation-analysis.ipynb`](notebooks/1.2-mw-correlation-analysis.ipynb) - Additional correlation studies
- [`1.0-mw-clustering.ipynb`](notebooks/1.0-mw-clustering.ipynb) - Building clustering analysis
- [`1.0-mw-data-cleaning.ipynb`](notebooks/1.0-mw-data-cleaning.ipynb) - Data preprocessing and cleaning

#### Model Development
- [`1.0-mw-darts-test.ipynb`](notebooks/1.0-mw-darts-test.ipynb) - Testing Darts forecasting library
- [`1.0-mw-fcn-training.ipynb`](notebooks/1.0-mw-fcn-training.ipynb) - Fully Connected Network training
- [`1.0-mw-hf-transformers.ipynb`](notebooks/1.0-mw-hf-transformers.ipynb) - Hugging Face Transformers experiments

#### Data Source Notebooks
- [`1.0-mw-dh-data.ipynb`](notebooks/1.0-mw-dh-data.ipynb) - District heating data analysis
- [`1.0-mw-kinergy-data.ipynb`](notebooks/1.0-mw-kinergy-data.ipynb) - Kinergy data processing
- [`1.0-mw-data-exploration-dh-kinergy.ipynb`](notebooks/1.0-mw-data-exploration-dh-kinergy.ipynb) - Combined DH and Kinergy data exploration

#### Feature Engineering
- [`1.0-mw-features.ipynb`](notebooks/1.0-mw-features.ipynb) - Feature engineering experiments
- [`webfactor_time_series_forecasting.ipynb`](notebooks/webfactor_time_series_forecasting.ipynb) - Web factor time series analysis

### References Directory ([`references/`](references))

Configuration files and metadata:

- [`configs.jsonl`](references/configs.jsonl) - Model configuration parameters in JSON Lines format
- [`liegenschaften_missing_qm_wohnung.csv`](references/liegenschaften_missing_qm_wohnung.csv) - Building metadata with missing area information

### Reports Directory ([`reports/`](reports))

Generated analysis results and visualizations:

#### Evaluation Results
- [`reduced_data_eval_1_day_forecast.csv`](reports/reduced_data_eval_1_day_forecast.csv) - 1-day forecast evaluation results
- [`reduced_data_eval_3_hour_forecast.csv`](reports/reduced_data_eval_3_hour_forecast.csv) - 3-hour forecast evaluation results
- [`reduced_data_eval_7_day_forecast.csv`](reports/reduced_data_eval_7_day_forecast.csv) - 7-day forecast evaluation results
- [`reduced_data_eval_24_hour_forecast.csv`](reports/reduced_data_eval_24_hour_forecast.csv) - 24-hour forecast evaluation results
- [`results_cluster_eval.csv`](reports/results_cluster_eval.csv) - Cluster-based evaluation results

#### Figures ([`reports/figures/`](reports/figures/))
- [`daily_time_span.png`](reports/figures/daily_time_span.png) - Daily data time span visualization
- [`hourly_time_span.png`](reports/figures/hourly_time_span.png) - Hourly data time span visualization
- [`metrics_per_step.png`](reports/figures/metrics_per_step.png) - Multi-step forecast metrics
- [`reduced_data_eval_1_day_forecast.png`](reports/figures/reduced_data_eval_1_day_forecast.png) - 1-day forecast visualization
- [`reduced_data_eval_7_day_forecast.png`](reports/figures/reduced_data_eval_7_day_forecast.png) - 7-day forecast visualization
- [`example_flat_lines/`](reports/figures/example_flat_lines/) - Examples of flat line patterns in data
- [`missing_data/hourly/`](reports/figures/missing_data/hourly/) - Visualizations of missing data patterns

### Source Code Directory ([`src/energy_forecast/`](src/energy_forecast))

Main source code package:

#### Core Files
- [`__init__.py`](src/energy_forecast/__init__.py) - Package initialization
- [`config.py`](src/energy_forecast/config.py) - Configuration constants and feature definitions
- [`dataset.py`](src/energy_forecast/dataset.py) - Data loading and dataset classes for creating data in `data/processed`
- [`features.py`](src/energy_forecast/features.py) - Feature engineering utilities
- [`plots.py`](src/energy_forecast/plots.py) - Visualization functions [created with Claude Sonnet 3.7]
- [`get_run_summary.py`](src/energy_forecast/get_run_summary.py) - Experiment run summary utilities for getting results from WandB
- [`sweep.py`](src/energy_forecast/sweep.py) - Hyperparameter sweeping utilities

#### Data Processing ([`src/energy_forecast/data_processing/`](src/energy_forecast/data_processing/))
- [`__init__.py`](src/energy_forecast/data_processing/__init__.py) - Module initialization
- [`data_loader.py`](src/energy_forecast/data_processing/data_loader.py) - Data loading utilities
- [`feature_generation.py`](src/energy_forecast/data_processing/feature_generation.py) - Feature creation and engineering
- [`lod_data_processor.py`](src/energy_forecast/data_processing/lod_data_processor.py) - LOD (Level of Detail) data processing
- [`ou_process.py`](src/energy_forecast/data_processing/ou_process.py) - Ornstein-Uhlenbeck process implementation
- [`preprocessing.py`](src/energy_forecast/data_processing/preprocessing.py) - Data preprocessing pipeline
- [`swn_api.py`](src/energy_forecast/data_processing/swn_api.py) - SWN API integration

#### Model Implementations ([`src/energy_forecast/model/`](src/energy_forecast/model))
- [`__init__.py`](src/energy_forecast/model/__init__.py) - Module initialization
- [`train.py`](src/energy_forecast/model/train.py) - Main training pipeline and script
- [`models.py`](src/energy_forecast/model/models.py) - Model class definitions and implementations
- [`evaluate.py`](src/energy_forecast/model/evaluate.py) - Model evaluation utilities
- [`predict.py`](src/energy_forecast/model/predict.py) - Prediction utilities
- [`transformer.py`](src/energy_forecast/model/transformer.py) - Transformer model implementation
- [`xlstm.py`](src/energy_forecast/model/xlstm.py) - xLSTM model implementation
- [`tft_xlstm_fusion.py`](src/energy_forecast/model/tft_xlstm_fusion.py) - TFT-xLSTM fusion model [created with Claude Sonnet 3.7]
- [`darts_models.py`](src/energy_forecast/model/darts_models.py) - Darts library model wrappers

#### Temporal Fusion Transformer ([`src/energy_forecast/tft/`](src/energy_forecast/tft/))
Complete TFT implementation with:
- [`README.md`](src/energy_forecast/tft/README.md) - TFT-specific documentation
- [`requirements.txt`](src/energy_forecast/tft/requirements.txt) - TFT dependencies
- [`run.sh`](src/energy_forecast/tft/run.sh) - TFT execution script

##### Data Formatters ([`src/energy_forecast/tft/data_formatters/`](src/energy_forecast/tft/data_formatters/))
- [`heat.py`](src/energy_forecast/tft/data_formatters/heat.py) - Heat data formatter
- [`heat_diff.py`](src/energy_forecast/tft/data_formatters/heat_diff.py) - Heat difference data formatter
- [`heat_hourly.py`](src/energy_forecast/tft/data_formatters/heat_hourly.py) - Hourly heat data formatter
- [`heat_no_building.py`](src/energy_forecast/tft/data_formatters/heat_no_building.py) - Heat data without building features

##### Experiment Settings ([`src/energy_forecast/tft/expt_settings/`](src/energy_forecast/tft/expt_settings/))
- [`configs.py`](src/energy_forecast/tft/expt_settings/configs.py) - TFT experiment configurations

##### Scripts
- [`script_download_data.py`](src/energy_forecast/tft/script_download_data.py) - Data download utility
- [`script_evaluate_fixed_params.py`](src/energy_forecast/tft/script_evaluate_fixed_params.py) - Fixed parameter evaluation
- [`script_hyperparam_opt.py`](src/energy_forecast/tft/script_hyperparam_opt.py) - Hyperparameter optimization
- [`script_train_fixed_params.py`](src/energy_forecast/tft/script_train_fixed_params.py) - Fixed parameter training

## Evaluation

Models are evaluated using multiple metrics:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **nRMSE**: Normalized RMSE

Results are saved in the `reports/` directory and logged to wandb if enabled. Models are stored in `models/` folder.

## Dependencies

Key dependencies include:
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: For xLSTM implementation
- **Polars/Pandas**: Data manipulation
- **Darts**: Time series forecasting library
- **Scikit-learn**: Machine learning utilities
- **Wandb**: Experiment tracking
- **Plotly/Matplotlib**: Visualization

## License

This project follows the Cookiecutter Data Science template structure and is intended for research and educational purposes.

## Acknowledgments

- Built using the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template
- Temporal Fusion Transformer implementation from the [official repository](https://github.com/google-research/google-research/tree/master/tft) by Google Research 
- xLSTM architecture implementation for enhanced sequence modeling from [official repository](https://github.com/NX-AI/xlstm)
- README.md created by [Claude Code](https://www.anthropic.com/claude-code)
- many helper functions, like saving and loading models, files, ... were created using the help of Claude Sonnet 3.7
- extensive function documentations were created with Claude Sonnet 3.5