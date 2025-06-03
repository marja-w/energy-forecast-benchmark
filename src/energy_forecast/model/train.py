import itertools

import jsonlines
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import os

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
    from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90, TrainDatasetBuilding
    from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
    NNModel, \
    RNN1Model, FCN2Model, FCN3Model, Baseline, RNN3Model, TransformerModel, LSTMModel, xLSTMModel, xLSTMTSFModel
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

        from src.energy_forecast.plots import plot_means, plot_std, plot_train_val_test_split
        from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
        from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90, TrainDatasetBuilding
        from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
            NNModel, RNN1Model, FCN2Model, FCN3Model, Baseline, RNN3Model, LSTMModel
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
    elif config["model"] == "Transformer":
        return TransformerModel(config)
    elif config["model"] == "lstm":
        return LSTMModel(config)
    elif config["model"] == "xlstm":
        return xLSTMModel(config)
    elif config["model"] == "xlstm_tsf":
        return xLSTMTSFModel(config)
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
    ds, run_config = prepare_dataset(run_config)

    # get model and baseline
    m = get_model(run_config)
    baseline = Baseline(run_config)

    # train
    run = m.train_ds(ds, log=run_config["log"])

    # Evaluate the models
    m.evaluate(ds, run, log=run_config["log"], plot=run_config["plot"])
    baseline.evaluate(ds, run)

    # per_cluster_evaluation(baseline, ds, m, run)

    # save model on disk and in wandb
    m.save()
    return run


if __name__ == '__main__':
    configs_path = REFERENCES_DIR / "configs.jsonl"
    models = ["Transformer"]
    scalers = ["standard"]
    feature_codes = [12, 14, 13]
    neurons_list = [100]
    n_ins = [3, 7]
    n_outs = [1, 7]
    n_futures = [0, 1, 7]
    epochs_list = [40]
    config = {"project": "ma-wahl-forecast",
              "log": False,  # whether to log to wandb
              "plot": False, # whether to plot predictions
              "energy": "all",
              "res": "hourly",
              "interpolate": 1,
              "dataset": "building",  # building, meta, missing_data_90
              "model": "RNN1",
              "lag_in": 72,
              "lag_out": 72,
              "n_in": 12,
              "n_out": 3,
              "n_future": 3,
              "scaler": "standard",
              "scale_mode": "individual",  # all, individual
              "feature_code": 15,
              "train_test_split_method": "time",
              "epochs": 1,
              "optimizer": "adam",
              "loss": "mean_squared_error",
              "metrics": ["mae"],
              "batch_size": 64,
              "dropout": 0.1,
              "neurons": 100,
              "lr_scheduler": "none",  # none, step_decay
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
            wandb_run = train(config_dict)
            wandb_run.finish()  # finish run to start new run with next config
    elif all_models:
        for model, feature_code, n_in, n_out, n_f, epochs, neurons, scaler in itertools.product(models, feature_codes, n_ins, n_outs, n_futures, epochs_list, neurons_list, scalers):
            if n_f != n_out: continue
            if feature_code == 12 and n_f > 0: continue  # only feature is diff
            if model == "FCN3" and n_in == 1 and neurons != 120 and n_out == 1: continue
            if model == "FCN3" and feature_code == 12: continue
            logger.info(f"Training combination: feature code {feature_code}, n_in {n_in}, n_out {n_out}, n_future {n_f}, epochs {epochs}, neurons {neurons}, scaler {scaler}")
            config["model"] = model
            config["feature_code"] = feature_code
            config["n_in"] = n_in
            config["n_out"] = n_out
            config["n_future"] = n_f
            config["epochs"] = epochs
            config["neurons"] = neurons
            config["scaler"] = scaler
            try:
                del config["features"]  # delete feature names if set in previous run
            except KeyError:
                pass
            wandb_run = train(config)
            wandb_run.finish()
    else:
        wandb_run = train(config)
        if wandb_run: wandb_run.finish()
