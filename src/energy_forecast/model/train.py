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
from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, NNModel


def get_model(config: dict) -> Model:
    if config["model"] == "FCN1":
        return FCNModel(config)
    elif config["model"] == "DT":
        return DTModel(config)
    else:
        raise Exception(f"Unknown model {config['model']}")


def get_data(config: dict) -> tuple[pl.DataFrame, dict]:
    """
    Creates a Dataset and returns its polars.DataFrame. One-Hot-Encodes data and updates config with new feature names.
    Adds multiple forecast data if needed. Filters for energy type.
    Args:
        config: dictionary setting "n_out" and "energy" parameter, as well

    Returns:

    """
    ds = Dataset()
    ds.load_feat_data()  # all data
    config = ds.one_hot_encode(config)  # one hot encode categorical features
    ds.add_multiple_forecast(config["n_out"])  # add multiple steps if forecast larger than one
    df = ds.df
    # select energy type
    if config["energy"] != "all":
        df = df.filter(pl.col("primary_energy") == config["energy"])
    df = df.drop_nulls(subset=config["features"])  # remove null values for used features
    return df, config


def get_train_test_val_split(config: dict, df: pl.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_per = 0.8
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_per, random_state=42)
    df = df.with_row_index()
    for train_idx, test_idx in gss.split(df, groups=df["id"]):
        train_data = df.filter(pl.col("index").is_in(train_idx))
        test_val_df = df.filter(pl.col("index").is_in(test_idx))

    # split test into validation and test
    test_val_df = test_val_df.drop("index").with_row_index()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    for test_idx, val_idx in gss.split(test_val_df, groups=test_val_df["id"]):
        test_data = test_val_df.filter(pl.col("index").is_in(test_idx))
        val_data = test_val_df.filter(pl.col("index").is_in(val_idx))

    # transform to pandas DataFrame input
    target_vars = ["diff"]
    if config["n_out"] > 1:
        target_vars += [f"diff_t+{i}" for i in range(1, config["n_out"])]
    X_train = train_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    y_train = train_data.to_pandas()[target_vars]
    X_val = val_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    y_val = val_data.to_pandas()[target_vars]
    X_test: pd.DataFrame = test_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    y_test: pd.DataFrame = test_data.to_pandas()[target_vars]

    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    return X_train, X_test, X_val, y_train, y_test, y_val


def train(config: dict):
    # Load the data
    df, config = get_data(config)

    # train test split
    df_train, df_test, df_val, df_y_train, df_y_test, df_y_val = get_train_test_val_split(config, df)

    # get model and baseline
    m = get_model(config)
    baseline = LinearRegressorModel(config)

    # train
    model, run = m.train(df_train, df_y_train, df_val, df_y_val)
    baseline.fit(df_train, df_y_train)

    # Evaluate the models
    baseline.evaluate(df_test, df_y_test, run)
    m.evaluate(df_test, df_y_test, run)

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
        for config in reader.iter():
            configs.append(config)

    for config in tqdm(configs):  # start one training for each config
        _, run = train(config)
        run.finish()  # finish run to start new run with next config
