import os
from pathlib import Path
import tensorflow as tf

import wandb
from typing_extensions import LiteralString
from wandb.apis.importers.internals.internal import ROOT_DIR

from src.energy_forecast.config import MODELS_DIR, PROJ_ROOT, FEATURE_SETS
from src.energy_forecast.model.train import prepare_dataset, get_model


def download_from_wandb(run_id: str) -> LiteralString | str | bytes:
    # Initialize the API
    api = wandb.Api()
    # Get the run details
    run_path = f"rausch-technology/ma-wahl-forecast/{run_id}"
    run = api.run(run_path)
    print(f"Connected to run: {run_path}")
    # Create a directory to store downloaded model files if it doesn't exist
    download_dir = PROJ_ROOT
    # Iterate through the run's files and download those in the "model" folder
    for file in run.files():
        if file.name.startswith("models/"):
            print(f"Downloading file: {file.name}")
            path_to_file = os.path.join(download_dir, file.name)
            if os.path.exists(path_to_file):
                os.remove(path_to_file)
            file.download(root=download_dir, replace=True)
            break
    print(f"Download complete. Files are saved in: {MODELS_DIR}")
    run = wandb.init(entity="rausch-technology", project="ma-wahl-forecast", id=run_id, resume="allow",
                     allow_val_change=True)
    return run, path_to_file


def get_model_from_wandb(run_id: str):
    run, path_to_model_file = download_from_wandb(run_id)
    orig_features = run.config["features"]
    run.config.update({"features": FEATURE_SETS[run.config["feature_code"]]}, allow_val_change=True)
    ds, config = prepare_dataset(run.config)
    run.config.update({"features": orig_features}, allow_val_change=True)
    # load model
    m = get_model(config)
    m.load_model_from_file(path_to_model_file)
    return m, ds, run


def get_model_from_path(path_to_model: Path, config: dict):
    # get dataset
    config["features"] = FEATURE_SETS[config["feature_code"]]
    ds, config = prepare_dataset(config)
    m = get_model(config)
    m.load_model_from_file(path_to_model)
    return m, ds


def main(run_id: str):
    m, ds, run = get_model_from_wandb(run_id=run_id)
    # evaluate and plot predictions
    m.evaluate(ds, run, log=False, plot=True)
    run.finish()


def main_local(path_to_model: Path, config: dict):
    m, ds = get_model_from_path(path_to_model, config)
    m.evaluate(ds, None, log=True, plot=False)


if __name__ == '__main__':
    run_id = "62ne9qky"
    training_config = {
        "energy": "all",
        "res": "daily",
        "interpolate": 1,
        "dataset": "building",  # building, meta, missing_data_90
        "model": "tft",
        "lag_in": 7,
        "lag_out": 7,
        "n_in": 7,
        "n_out": 7,
        "n_future": 0,
        "scaler": "standard",
        "scale_mode": "individual",  # all, individual
        "feature_code": 14,
        "train_test_split_method": "time",
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "metrics": ["mae"],
        "batch_size": 32,
        "dropout": 0.1,
        "neurons": 100,
        "lr_scheduler": "none",
        "weight_initializer": "glorot",
        "activation": "relu"
    }
    model_path = MODELS_DIR / "heat" / "fixed"
    # main_local(model_path, training_config)
    main(run_id)
