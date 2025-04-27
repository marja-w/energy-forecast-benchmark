import itertools

import jsonlines
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import os

from src.energy_forecast.model.darts_models import DartsTransformer, DartsRNN, DartsBlockRNN, DartsLSTM
from src.energy_forecast.utils.train_test_val_split import get_train_test_val_split
from src.energy_forecast.utils.util import store_df_wandb

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
    from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90, TimeSeriesDataset, \
        TrainDatasetBuilding
    from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
        NNModel, \
        RNN1Model, FCN2Model, FCN3Model, Baseline, RNN3Model, TransformerModel, LSTMModel
except ModuleNotFoundError:
    import sys
    import os
    import inspect

    curr_frame = inspect.currentframe()
    if curr_frame:
        curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(curr_frame)))
        par_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
        sys.path.insert(0, par_dir)

        from src.energy_forecast.plots import plot_means, plot_std, plot_train_val_test_split
        from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS
        from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90
        from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
            NNModel, RNN1Model, FCN2Model, FCN3Model, Baseline, DartsModel, RNN3Model
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
    elif config["model"] == "RNN2":
        return DartsRNN(config)
    elif config["model"] == "RNN3":
        return RNN3Model(config)
    elif config["model"] == "Transformer":
        return TransformerModel(config)
    elif config["model"] == "lstm":
        return LSTMModel(config)
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
        case _:
            logger.warning(f"Unknown dataset {config['dataset']}. Using default dataset TrainingDataset.")
            ds = TrainingDataset(config)
    interpolate = config["interpolate"]
    ds.load_feat_data(bool(interpolate))  # all data
    ds.preprocess()  # preprocess data for training
    return ds


def get_features(code: int):
    return FEATURE_SETS[code]


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


def prepare_dataset(run_config: dict) -> TrainingDataset:
    try:
        logger.info(f"Features: {run_config['features']}")
    except KeyError:
        run_config["features"] = get_features(run_config["feature_code"])
    # Load the data
    ds = get_data(run_config)
    # train test split
    ds = get_train_test_val_split(ds)
    # scaling
    ds.fit_scalers()
    return ds, run_config


def train(config: dict):
    ds, config = prepare_dataset(config)

    # get model and baseline
    m = get_model(config)
    baseline = Baseline(config)

    # train
    run = m.train_ds(ds)

    # Evaluate the models
    m.evaluate_ds(ds, run)
    baseline.evaluate(ds, run)

    # per_cluster_evaluation(baseline, ds, m, run)

    # save model on disk and in wandb
    m.save()
    return run


if __name__ == '__main__':
    configs_path = REFERENCES_DIR / "configs.jsonl"
    models = ["RNN1", "lstm", "Transformer"]
    feature_codes = [12, 14, 13]
    n_ins = [1, 7]
    n_outs = [1, 7]
    config = {"project": "ma-wahl-forecast",
              "energy": "all",
              "res": "daily",
              "interpolate": 1,
              "dataset": "building",
              "model": "FCN3",
              "train_len": 32,
              "lag_in": 7,
              "lag_out": 7,
              "n_in": 7,
              "n_out": 7,
              "n_future": 7,
              "scaler": "standard",
              "scale_mode": "all",  # all, individual
              "feature_code": 14,
              "train_test_split_method": "time",
              "epochs": 40,
              "optimizer": "adam",
              "loss": "mean_squared_error",
              "metrics": ["mae"],
              "batch_size": 32,
              "dropout": 0.1,
              "neurons": 70,
              "lr_scheduler": "none",
              "weight_initializer": "glorot",
              "activation": "relu"}  # ReLU, Linear
    # config = None
    all_models = False
    if config is None:
        # Read in configs from .jsonl file
        configs = list()
        with jsonlines.open(configs_path) as reader:
            for config_dict in reader.iter():
                configs.append(config_dict)

        for config_dict in tqdm(configs):  # start one training for each config
            run = train(config_dict)
            run.finish()  # finish run to start new run with next config
    elif all_models:
        for model, feature_code, n_in, n_out in itertools.product(models, feature_codes, n_ins, n_outs):
            config["model"] = model
            config["feature_code"] = feature_code
            config["n_in"] = n_in
            config["n_out"] = n_out
            run = train(config)
            run.finish()
    else:
        run = train(config)
        run.finish()
