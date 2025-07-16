import argparse
import json
from dataclasses import dataclass
from typing import Any

import jsonlines
import polars as pl
import wandb
from loguru import logger
from tqdm import tqdm

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
    from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90, TrainDatasetBuilding, \
        TrainDatasetNoise
    from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
        NNModel, \
        RNN1Model, FCN2Model, FCN3Model, Baseline, RNN3Model, TransformerModel, LSTMModel, xLSTMModel, xLSTMModel_v2
    from src.energy_forecast.utils.train_test_val_split import get_train_test_val_split
    from src.energy_forecast.utils.util import store_df_wandb
except ModuleNotFoundError:
    import sys
    import os
    import inspect

    curr_frame = inspect.currentframe()
    if curr_frame:
        curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(curr_frame)))
        par_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
        sys.path.insert(0, par_dir)
        logger.info(par_dir)

        from src.energy_forecast.plots import plot_means, plot_std
        from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
        from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90, TrainDatasetBuilding, \
            TrainDatasetNoise
        from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
            NNModel, RNN1Model, FCN2Model, FCN3Model, Baseline, RNN3Model, LSTMModel, TransformerModel, xLSTMModel, \
            xLSTMModel_v2
        from src.energy_forecast.utils.train_test_val_split import get_train_test_val_split
        from src.energy_forecast.utils.util import store_df_wandb
    else:
        raise IOError("Current Frame not found")


def get_model(config: dict) -> Model:
    if config["model"] == "FCN1":
        return FCNModel(config)
    elif config["model"] == "FCN2":
        return FCN2Model(config)
    elif config["model"] == "FCN3":
        return FCN3Model(config)
    elif config["model"] == "DT":
        return DTModel(config)
    elif config["model"] == "RNN1":
        return RNN1Model(config)
    elif config["model"] == "RNN3":
        return RNN3Model(config)
    elif config["model"] == "transformer":
        return TransformerModel(config)
    elif config["model"] == "lstm":
        return LSTMModel(config)
    elif config["model"] == "xlstm":
        return xLSTMModel(config)
    elif config["model"] == "xlstm-tft":
        return xLSTMModel_v2(config)
    else:
        raise Exception(f"Unknown model {config['model']}")


def get_data(config: dict) -> TrainingDataset:
    """
    Creates a Dataset and returns its polars.DataFrame. One-Hot-Encodes data and updates config with new feature names.
    Adds multiple forecast data if needed. Filters for energy type.
    Args:
        config: dictionary setting "n_out" and "energy" parameter, as well

    Returns:

    """
    match config["dataset"]:
        case "building":
            ds = TrainDatasetBuilding(config)
        case "meta":
            ds = TrainingDataset(config)
        case "missing_data_90":
            ds = TrainDataset90(config)
        case "building_noise":
            ds = TrainDatasetNoise(config)
        case _:
            logger.warning(f"Unknown dataset {config['dataset']}. Using default dataset TrainingDataset.")
            ds = TrainingDataset(config)
    interpolate = config["interpolate"]
    ds.load_feat_data(bool(interpolate))  # all data
    ds.preprocess()  # preprocess data for training
    return ds


@dataclass
class TrainConfig:
    """Configuration for model training"""
    # model specifics
    model: str
    n_in: int
    n_out: int
    n_future: int
    feature_code: int

    # training
    epochs: int
    batch_size: int
    dropout: float = 0.1
    neurons: int = 100
    num_heads: int = 4
    optimizer: str = "adam"
    loss: str = "mean_squared_error"
    remove_per: float = 0.0
    lr_scheduler: str = "none"
    weight_initializer: str = "glorot"
    activation: str = "relu"
    transformer_blocks: int = 2

    # will be overwritten
    lag_in: int = 7
    lag_out: int = 7
    metrics: list[str] = None

    # preprocessing
    scaler: str = "standard"
    scale_mode: str = "individual"
    train_test_split_method: str = "time"

    # wandb configuration
    project: str = "ma-wahl-forecast"
    log: bool = False
    plot: bool = False

    # dataset specifications
    energy: str = "all"
    res: str = "daily"
    interpolate: int = 1
    dataset: str = "building"

    def get(self, key: str, default: Any = None):
        return getattr(self, key, default)

    def as_dict(self) -> dict:
        return {
            'model': self.model,
            'lag_in': self.lag_in,
            'lag_out': self.lag_out,
            'n_in': self.n_in,
            'n_out': self.n_out,
            'n_future': self.n_future,
            'feature_code': self.feature_code,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'metrics': self.metrics,
            'dropout': self.dropout,
            'neurons': self.neurons,
            'num_heads': self.num_heads,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'remove_per': self.remove_per,
            'lr_scheduler': self.lr_scheduler,
            'weight_initializer': self.weight_initializer,
            'activation': self.activation,
            'transformer_blocks': self.transformer_blocks,
            'scaler': self.scaler,
            'scale_mode': self.scale_mode,
            'train_test_split_method': self.train_test_split_method,
            'project': self.project,
            'log': self.log,
            'plot': self.plot,
            'energy': self.energy,
            'res': self.res,
            'interpolate': self.interpolate,
            'dataset': self.dataset
        }


def get_train_config(run_config: dict) -> dict:
    run_config = TrainConfig(**run_config)
    if not run_config.get("metrics"):
        run_config.metrics = ["mae"]
    if run_config.get("res") == "daily":
        run_config.lag_in = run_config.lag_out = 7
    else:
        run_config.lag_in = run_config.lag_out = 72
    return run_config.as_dict()


def per_cluster_evaluation(baseline: Baseline, ds: TrainingDataset, m: Model,
                           wandb_run: wandb.sdk.wandb_run.Run) -> None:
    clusters = ds.compute_clusters()
    eval_dict_m = m.evaluate_per_cluster(ds.X_test, ds.y_test, wandb_run, clusters)
    eval_dict_b = baseline.evaluate_per_cluster(ds, wandb_run, clusters)
    eval_df = pl.concat([pl.DataFrame(eval_dict_m), pl.DataFrame(eval_dict_b)], how="horizontal")
    file_name = "results_cluster_eval.csv"
    eval_df.write_csv(REPORTS_DIR / file_name)
    # save to wandb run
    store_df_wandb(eval_df, "results_cluster_eval.txt")


def prepare_dataset(run_config: dict) -> tuple[TrainingDataset, dict]:
    """
    Prepare the dataset for training or evaluation. Loads data from disk according to run_config.
    Splits data into train, val, and test sets, using the specified method in run_config. Fits scalers to the train set
    and stores the transformed data in the TrainingDataset.
    :param run_config: needs to specifiy "dataset" (building, meta, missing_data_90), "n_out"
    (number of forecast steps), "interpolate" (whether to use interpolated data), "energy" (which energy type to use),
    "train_test_split_method" (how to split the data), "scaler" (which scaler to use), "scale_mode" (whether to scale on
    all target variables or on individual series)
    :return: the training dataset and the run_config dictionary with updated values for "features"
    """
    # Load the data
    ds = get_data(run_config)
    # train test split
    ds = get_train_test_val_split(ds)
    # scaling
    ds.fit_scalers()
    ds.config["features"].sort()
    return ds, ds.config


def train(run_config: dict):
    run_config = get_train_config(run_config)
    ds, run_config = prepare_dataset(run_config)

    # get model and baseline
    m = get_model(run_config)
    baseline = Baseline(run_config)

    # train
    run = m.train_ds(ds, log=run_config["log"])
    ds.compute_rmse_noisy_features()

    # Evaluate the models
    m.evaluate(ds, run, log=run_config["log"], plot=run_config["plot"])
    baseline.evaluate(ds, run)

    # per_cluster_evaluation(baseline, ds, m, run)

    # save model on disk and in wandb
    m.save()
    return run


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process a config file specified via command line.')

    # Add the batch_file argument
    parser.add_argument('--config_file', type=str, required=False,
                        help='Path to the batch file to be processed')

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the batch_file parameter
    config_file_name = args.config_file

    if config_file_name is not None:
        logger.info(f"The specified batch file is: {config_file_name}")
        configs_path = REFERENCES_DIR / f"{config_file_name}.json"
        with open(configs_path, "r") as f:
            config = json.load(f)
        wandb_run = train(config)
        if wandb_run:
            wandb_run.finish()
    else:
        configs_path = REFERENCES_DIR / "configs.jsonl"
        # Read in configs from .jsonl file
        configs = list()
        with jsonlines.open(configs_path) as reader:
            for config_dict in reader.iter():
                configs.append(config_dict)

        for config_dict in tqdm(configs):  # start one training for each config
            wandb_run = train(config_dict)
            wandb_run.finish()  # finish run to start new run with next config
