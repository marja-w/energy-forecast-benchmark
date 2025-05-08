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
    run = wandb.init(entity="rausch-technology", project="ma-wahl-forecast", id=run_id, resume="allow", allow_val_change=True)
    return run, path_to_file


def get_model_from_wandb(run_id: str):
    run, path_to_model_file = download_from_wandb(run_id)
    # get dataset
    run.config.update({"features": FEATURE_SETS[run.config["feature_code"]]}, allow_val_change=True)
    config = run.config
    ds, config = prepare_dataset(config)
    # load model
    m = get_model(config)
    m.load_model_from_file(path_to_model_file)
    return m, ds, run


def main(run_id: str):
    m, ds, run = get_model_from_wandb(run_id=run_id)
    # evaluate and plot predictions
    m.evaluate(ds, run, log=False, plot=True)
    run.finish()

if __name__ == '__main__':
    run_id = "dyq4e0zk"
    main(run_id)
