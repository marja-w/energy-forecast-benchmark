import pandas as pd
import tensorflow as tf
import jsonlines
import wandb
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import polars as pl
from tqdm import tqdm

from src.energy_forecast.config import REFERENCES_DIR
from src.energy_forecast.dataset import Dataset
from src.energy_forecast.model.models import Model, FCNModel, LinearRegressorModel


def get_model(config: dict) -> Model:
    if config["model"] == "FCN1":
        return FCNModel(config)
    else:
        raise Exception(f"Unknown model {config['model']}")


def get_data(config: dict) -> Dataset:
    ds = Dataset()
    ds.load_feat_data()  # all data
    df = ds.df
    # select energy type
    if config["energy"] != "all":
        df = df.filter(pl.col("primary_energy") == config["energy"])
    df = df.drop_nulls(subset=config["features"])  # remove null values for used features
    return df


def train(config: dict):
    # Load the data
    df = get_data(config)

    # train test split
    train_per = 0.8
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_per, random_state=42)
    df = df.with_row_index()
    for train_idx, test_idx in gss.split(df, groups=df["id"]):
        train_data = df.filter(pl.col("index").is_in(train_idx))
        test_data = df.filter(pl.col("index").is_in(test_idx))

    # transform to pandas DataFrame input
    X_train = train_data.to_pandas()[list(set(config["features"]) - {"diff"})]
    y_train = train_data.to_pandas()["diff"]

    X_test: pd.DataFrame = test_data.to_pandas()[list(set(config["features"]) - {"diff"})]
    y_test: pd.DataFrame = test_data.to_pandas()["diff"]

    # get model and baseline
    m = get_model(config)
    baseline = LinearRegressorModel(config)

    # train
    model, run = m.train(X_train, y_train, X_test, y_test)
    baseline.fit(X_train, y_train)

    # Evaluate the model
    test_loss, test_mae, test_nrmse = m.evaluate(X_test, y_test)
    b_nrmse = baseline.evaluate(X_test, y_test)
    logger.info(
        f"MSE Loss on test data: {test_loss}, RMSE Loss on test data: {test_mae}"
    )
    run.log(data={"test_mse": test_loss, "test_mae": test_mae, "b_nrmse": b_nrmse})

    # save model on disk and in wandb
    model_path = m.save()
    wandb.save(model_path)
    return m, run


if __name__ == '__main__':
    configs_path = REFERENCES_DIR / "configs.jsonl"
    attributes = ["diff", "diff_t-1", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
                  'wspd', 'wpgt',
                  'pres', 'tsun', "holiday"]
    attributes_ha = attributes + ["heated_area", "anzahlwhg"]

    # Read in configs from .jsonl file
    configs = list()
    with jsonlines.open(configs_path) as reader:
        for config in reader.iter():
            configs.append(config)

    for config in tqdm(configs):  # start one training for each config
        _, run = train(config)
        run.finish()  # finish run to start new run with next config
