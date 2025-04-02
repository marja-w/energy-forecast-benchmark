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

from src.energy_forecast.utils.train_test_val_split import get_train_test_val_split

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
    from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90
    from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
        NNModel, \
        RNN1Model, FCN2Model, FCN3Model, Baseline, DartsModel
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
            NNModel, RNN1Model, FCN2Model, FCN3Model, Baseline
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
        return DartsModel(config)
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
    try:
        if config["missing_data"] == 90:
            ds = TrainDataset90(config)
        else:
            ds = TrainingDataset(config)
    except KeyError:
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
    eval_df.write_csv(REPORTS_DIR / "results_cluster_eval.csv")
    # save to wandb run
    # copy to wandb run dir to fix symlink issue
    os.makedirs(os.path.join(wandb.run.dir, "reports"))
    wandb_run_dir_eval = os.path.join(wandb.run.dir, os.path.join("reports", "results_cluster_eval.txt"))
    eval_df.to_pandas().to_csv(wandb_run_dir_eval, header=True, index=None, sep="\t", mode="a")
    wandb.save(wandb_run_dir_eval)


def train(config: dict):
    try:
        logger.info(f"Features: {config['features']}")
    except KeyError:
        config["features"] = get_features(config["feature_code"])
    # Load the data
    ds = get_data(config)

    # train test split
    ds = get_train_test_val_split(ds)

    # get model and baseline
    m = get_model(config)
    baseline = Baseline(config)

    # train
    model, run = m.train(ds)

    # Evaluate the models
    X_test_copy = ds.X_test.copy()
    y_test_copy = ds.y_test.copy()
    m.evaluate_ds(ds, run)
    assert X_test_copy.equals(ds.X_test)
    assert y_test_copy.equals(ds.y_test)
    baseline.evaluate(ds, run)
    assert X_test_copy.equals(ds.X_test)
    assert y_test_copy.equals(ds.y_test)

    per_cluster_evaluation(baseline, ds, m, run)

    # save model on disk and in wandb
    m.save()
    return m, run


if __name__ == '__main__':
    configs_path = REFERENCES_DIR / "configs.jsonl"
    attributes = ["diff", "diff_t-1"]
    attributes_weather = ['hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
                          'wspd', 'wpgt', 'pres', 'tsun']
    attributes_building = ["daily_avg", "heated_area", "anzahlwhg", "typ"]
    attributes_time = ["weekend", "holiday"]
    attributes_dh = ["ground_surface", "building_height", "storeys_above_ground"]
    # Read in configs from .jsonl file
    configs = list()
    with jsonlines.open(configs_path) as reader:
        for config_dict in reader.iter():
            configs.append(config_dict)

    for config_dict in tqdm(configs):  # start one training for each config
        _, run = train(config_dict)
        run.finish()  # finish run to start new run with next config
