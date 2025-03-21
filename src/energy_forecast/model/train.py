import jsonlines
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS
    from src.energy_forecast.dataset import Dataset, TrainingDataset
    from src.energy_forecast.model.models import Model, FCNModel, DTModel, LinearRegressorModel, RegressionModel, \
        NNModel, \
        RNN1Model, FCN2Model, FCN3Model, Baseline
except ModuleNotFoundError:
    import sys
    import os
    import inspect

    curr_frame = inspect.currentframe()
    if curr_frame:
        curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(curr_frame)))
        par_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_dir)))
        sys.path.insert(0, par_dir)

        from src.energy_forecast.plots import plot_means, plot_std
        from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS
        from src.energy_forecast.dataset import Dataset, TrainingDataset
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
    ds = TrainingDataset(config)
    ds.load_feat_data()  # all data
    df, config = ds.preprocess()  # preprocess data for training
    return df, config


def train_test_split_group_based(df, train_per):
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

    return train_data, val_data, test_data


def train_test_split_time_based(df: pl.DataFrame, train_per: float):
    df = df.sort([pl.col("id"), pl.col("datetime")])

    # Group by building index and split
    train_dfs = []
    val_test_dfs = []
    test_dfs = []
    val_dfs = []

    for group in df.group_by(pl.col("id")):
        building_df = group[1]  # Extract grouped DataFrame
        building_df = building_df.with_row_index()
        split_idx = int(len(building_df) * train_per)

        train_dfs.append(building_df.filter(pl.col("index") <= split_idx).drop(["index"]))
        val_test_dfs.append(building_df.filter(pl.col("index") > split_idx).drop(["index"]))

    for building_df in val_test_dfs:
        building_df = building_df.with_row_index()
        split_idx = int(len(building_df) * 0.5)

        val_dfs.append(building_df.filter(pl.col("index") <= split_idx).drop(["index"]))
        test_dfs.append(building_df.filter(pl.col("index") > split_idx).drop(["index"]))

    # Concatenate results
    train_df = pl.concat(train_dfs)
    val_df = pl.concat(val_dfs)
    test_df = pl.concat(test_dfs)
    assert len(df) == (len(train_df) + len(val_df) + len(test_df))
    return train_df, val_df, test_df


def get_train_test_val_split(config: dict, df: pl.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train-test-val split for data. Ratio is 0.8/0.1/0.1
    :param config: setting "n_out" parameter for creating correct feature name list
    :param df: data
    :return: train-test-val split for data
    """
    train_per = 0.8
    try:
        split_method = config["train_test_split_method"]
    except KeyError:
        config["train_test_split_method"] = "time"  # default
    if config["train_test_split_method"] == "group":
        train_data, val_data, test_data = train_test_split_group_based(df, train_per)
    elif config["train_test_split_method"] == "time":
        train_data, val_data, test_data = train_test_split_time_based(df, train_per)

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

    # plot_means(X_train, y_train, X_val, y_val, X_test, y_test)
    # plot_std(X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_features(code: int):
    return FEATURE_SETS[code]


def train(config: dict):
    try:
        logger.info(f"Features: {config['features']}")
    except KeyError:
        config["features"] = get_features(config["feature_code"])
    # Load the data
    df, config = get_data(config)

    # train test split
    df_train, df_test, df_val, df_y_train, df_y_test, df_y_val = get_train_test_val_split(config, df)

    # get model and baseline
    m = get_model(config)
    baseline = Baseline(config)

    # train
    model, run = m.train(df_train, df_y_train, df_val, df_y_val)

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
        for config_dict in reader.iter():
            configs.append(config_dict)

    for config_dict in tqdm(configs):  # start one training for each config
        _, run = train(config_dict)
        run.finish()  # finish run to start new run with next config
