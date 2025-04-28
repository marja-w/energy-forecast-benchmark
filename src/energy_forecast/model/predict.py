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
            file.download(root=download_dir, replace=True)
            path_to_file = os.path.join(download_dir, file.name)
    print(f"Download complete. Files are saved in: {MODELS_DIR}")
    return run, path_to_file


if __name__ == '__main__':
    run, path_to_model_file = download_from_wandb(run_id="c19tm14i")

    # get dataset
    config = run.config
    config["features"] = FEATURE_SETS[config["feature_code"]]
    ds, config = prepare_dataset(config)

    # load model
    m = get_model(config)
    m.load_model_from_file(path_to_model_file)

    # evaluate and plot predictions
    m.evaluate_ds(ds, run, log=False, plot=True)
