from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
import polars as pl
from src.energy_forecast.dataset import TrainingDataset


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
        if len(train_b_df) <= (ds.config["train_len"]):
            logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")
            continue  # if series is too short, discard

        test_b_df = b_df.filter((pl.col("b_idx") > split_idx).and_(pl.col("b_idx") <= split_idx_two)).drop(["b_idx"])
        if len(test_b_df) <= (ds.config["n_in"]):
            logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")
            continue  # if series is too short, discard
        val_b_df = b_df.filter(pl.col("b_idx") > split_idx_two).drop(["b_idx"])
        if len(val_b_df) <= (ds.config["n_in"]):
            logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")

        train_dfs.append(train_b_df)
        test_dfs.append(test_b_df)
        val_dfs.append(val_b_df)

        ds.train_idxs.extend(train_b_df["index"].to_list())
        ds.test_idxs.extend(test_b_df["index"].to_list())
        ds.val_idxs.extend(val_b_df["index"].to_list())

    # sort indexes
    ds.train_idxs = sorted(ds.train_idxs)
    ds.test_idxs = sorted(ds.test_idxs)
    ds.val_idxs = sorted(ds.val_idxs)

    # Concatenate results
    train_df = pl.concat(train_dfs).drop("index")
    val_df = pl.concat(val_dfs).drop("index")
    test_df = pl.concat(test_dfs).drop("index")
    # plot_train_val_test_split(train_df, val_df, test_df)
    assert len(df) >= (len(train_df) + len(val_df) + len(test_df))  # might be smaller, because of discarded series
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
        target_vars += [f"diff(t+{i})" for i in range(1, config["n_out"])]

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
