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

try:
    from src.energy_forecast.plots import plot_means, plot_std
    from src.energy_forecast.config import REFERENCES_DIR, FEATURE_SETS, PROCESSED_DATA_DIR, REPORTS_DIR, N_CLUSTER
    from src.energy_forecast.dataset import Dataset, TrainingDataset, TrainDataset90
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


def train_test_split_group_based(ds: TrainingDataset, train_per: float) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = ds.df
    df = df.sort([pl.col("id"), pl.col("datetime")])
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_per, random_state=42)
    df = df.with_row_index()
    for train_idx, test_idx in gss.split(df, groups=df["id"]):
        train_data = df.filter(pl.col("index").is_in(train_idx))
        test_val_df = df.filter(pl.col("index").is_in(test_idx))
        ds.train_idxs = list(train_idx)
        ds.test_idxs = test_idx
    # split test into validation and test
    test_val_df = test_val_df.drop("index").with_row_index()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    for test_idx, val_idx in gss.split(test_val_df, groups=test_val_df["id"]):
        test_data = test_val_df.filter(pl.col("index").is_in(test_idx))
        val_data = test_val_df.filter(pl.col("index").is_in(val_idx))
        ds.val_idxs = list(ds.test_idxs[val_idx])
        ds.test_idxs = list(ds.test_idxs[test_idx])

    return train_data, val_data, test_data


def train_test_split_time_based(ds: TrainingDataset, train_per: float) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = ds.df
    df = df.sort([pl.col("id"), pl.col("datetime")])
    df = df.with_row_index()

    # Group by building index and split
    train_dfs = []
    test_dfs = []
    val_dfs = []

    for group in df.group_by(pl.col("id")):
        b_df = group[1]  # Extract grouped DataFrame
        b_df = b_df.with_row_index("b_idx")
        split_idx = int(len(b_df) * train_per)
        split_idx_two = int(len(b_df) * (((1 - train_per) / 2) + train_per))

        train_b_df = b_df.filter(pl.col("b_idx") <= split_idx).drop(["b_idx"])
        test_b_df = b_df.filter((pl.col("b_idx") > split_idx).and_(pl.col("b_idx") <= split_idx_two)).drop(["b_idx"])
        val_b_df = b_df.filter(pl.col("b_idx") > split_idx_two).drop(["b_idx"])

        train_dfs.append(train_b_df)
        test_dfs.append(test_b_df)
        val_dfs.append(val_b_df)

        ds.train_idxs.extend(train_b_df["index"].to_list())
        ds.test_idxs.extend(test_b_df["index"].to_list())
        ds.val_idxs.extend(val_b_df["index"].to_list())

    # Concatenate results
    train_df = pl.concat(train_dfs).drop("index")
    val_df = pl.concat(val_dfs).drop("index")
    test_df = pl.concat(test_dfs).drop("index")
    # plot_train_val_test_split(train_df, val_df, test_df)
    assert len(df) == (len(train_df) + len(val_df) + len(test_df))
    return train_df, val_df, test_df


def get_train_test_val_split(ds: TrainingDataset) -> TrainingDataset:
    """
    Train-test-val split for data. Ratio is 0.8/0.1/0.1
    :param ds: training dataset object, storing config, setting "n_out" parameter for creating correct feature name list
    :return: train-test-val split for data
    """
    train_per = 0.8
    config = ds.config
    try:
        split_method = config["train_test_split_method"]
    except KeyError:
        config["train_test_split_method"] = "time"  # default
    if config["train_test_split_method"] == "group":
        train_data, val_data, test_data = train_test_split_group_based(ds, train_per)
    elif config["train_test_split_method"] == "time":
        train_data, val_data, test_data = train_test_split_time_based(ds, train_per)

    # transform to pandas DataFrame input
    target_vars = ["diff"]
    if config["n_out"] > 1:
        target_vars += [f"diff_t+{i}" for i in range(1, config["n_out"])]

    train_data = train_data.sort([pl.col("id"), pl.col("datetime")])
    test_data = test_data.sort([pl.col("id"), pl.col("datetime")])
    val_data = val_data.sort([pl.col("id"), pl.col("datetime")])
    # selects only features and target variable
    ds.X_train = train_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    ds.y_train = train_data.to_pandas()[target_vars]
    ds.X_val = val_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    ds.y_val = val_data.to_pandas()[target_vars]
    ds.X_test = test_data.to_pandas()[list(set(config["features"]) - set(target_vars))]
    ds.y_test = test_data.to_pandas()[target_vars]

    # plot_means(X_train, y_train, X_val, y_val, X_test, y_test)
    # plot_std(X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info(f"Train data shape: {ds.X_train.shape}")
    logger.info(f"Test data shape: {ds.X_test.shape}")
    logger.info(f"Validation data shape: {ds.X_val.shape}")
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
    model, run = m.train(ds.X_train, ds.y_train, ds.X_val, ds.y_val)

    # Evaluate the models
    X_test_copy = ds.X_test.copy()
    y_test_copy = ds.y_test.copy()
    m.evaluate(ds.X_test, ds.y_test, run)
    assert X_test_copy.equals(ds.X_test)
    assert y_test_copy.equals(ds.y_test)
    assert X_test_copy.equals(ds.X_test)
    assert y_test_copy.equals(ds.y_test)
    baseline.evaluate(ds, run)
    assert X_test_copy.equals(ds.X_test)
    assert y_test_copy.equals(ds.y_test)

    per_cluster_evaluation(baseline, ds, m, run)

    # save model on disk and in wandb
    m.save()  # TODO: store cluster eval results here too
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
