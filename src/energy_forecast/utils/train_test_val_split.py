import datetime

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


def train_test_split_time_based(ds: TrainingDataset, train_per: float, remove_per: float=0.0) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = ds.df
    df = df.sort([pl.col("id"), pl.col("datetime")])
    df = df.with_row_index()
    ds.df = df

    # Group by building index and split
    train_dfs = []
    test_dfs = []
    val_dfs = []

    discarded_ids = list()
    for group in df.group_by(pl.col("id")):
        b_df = group[1]  # Extract grouped DataFrame
        b_df = b_df.with_row_index("b_idx")
        split_idx = int(len(b_df) * train_per)
        split_idx_two = int(len(b_df) * (((1 - train_per) / 2) + train_per))

        train_b_df = b_df.filter(pl.col("b_idx") <= split_idx)
        if remove_per != 0:
            # remove remove_per values of training data from beginning of series
            train_b_df = train_b_df.tail(int(len(train_b_df) * (1-remove_per)))
        train_b_df = train_b_df.drop(["b_idx"])

        try:
            lag_in, lag_out = ds.config["lag_in"], ds.config["lag_out"]
        except KeyError:
            lag_in, lag_out = ds.config["n_in"], ds.config["n_out"]
        min_len = lag_in + lag_out  # length needed for constructing one pair
        if len(train_b_df) <= min_len:  # if not one example can be produced from train series
            # logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")  # TODO dont throw away whole sensor
            discarded_ids.append(group[0][0])
            continue  # if series is too short, discard

        test_b_df = b_df.filter((pl.col("b_idx") > split_idx).and_(pl.col("b_idx") <= split_idx_two)).drop(["b_idx"])
        if len(test_b_df) <= min_len:
            # logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")
            discarded_ids.append(group[0][0])
            continue  # if series is too short, discard
        val_b_df = b_df.filter(pl.col("b_idx") > split_idx_two).drop(["b_idx"])
        if len(val_b_df) <= min_len:
            # logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")
            discarded_ids.append(group[0][0])
            continue
        train_dfs.append(train_b_df)
        test_dfs.append(test_b_df)
        val_dfs.append(val_b_df)

        ds.train_idxs.extend(train_b_df["index"].to_list())
        ds.test_idxs.extend(test_b_df["index"].to_list())
        ds.val_idxs.extend(val_b_df["index"].to_list())

    logger.info(f"Removed {len(discarded_ids)} series because they were too short")
    ds.discarded_ids = discarded_ids
    logger.info(f"Remaining series: {len(train_dfs)}")

    # sort indexes
    ds.train_idxs = sorted(ds.train_idxs)
    ds.test_idxs = sorted(ds.test_idxs)
    ds.val_idxs = sorted(ds.val_idxs)

    if remove_per == 0:  # only assert when no removals took place
        assert len(set(ds.df.filter(~(pl.col("id").is_in(ds.discarded_ids)))["index"].to_list()).symmetric_difference(
            set((ds.train_idxs + ds.test_idxs + ds.val_idxs)))) == 0  # check that all indexes that are used are present

    # Concatenate results
    train_df = pl.concat(train_dfs).drop("index")
    val_df = pl.concat(val_dfs).drop("index")
    test_df = pl.concat(test_dfs).drop("index")
    # plot_train_val_test_split(train_df, val_df, test_df)
    assert len(df) >= (len(train_df) + len(val_df) + len(test_df))  # might be smaller, because of discarded series
    return train_df, val_df, test_df


def train_test_split_date_based(ds: TrainingDataset, split_date: datetime.date, remove_per: float = 0.0) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits a given training dataset into training, validation,
    and testing sets. The splitting is performed
    by dividing the dataset into portions determined by the
    provided date.

    The function also sets the indexes of belonging to each split in the TrainingDataset and
    discards series if they are too short.

    :param remove_per: Percentage of train series that is discarded.
    :param ds: The dataset that needs to be split.
    :param split_date: A date indicating the point at which the data is split.
    :return: A tuple containing the training, validation, and testing
        datasets in the form of Polars DataFrames.
    """
    df = ds.df
    df = df.sort([pl.col("id"), pl.col("datetime")])
    df = df.with_row_index()

    # Group by building index and split
    train_dfs = []
    test_dfs = []
    val_dfs = []

    discarded_ids = list()
    train_per = 0.9 # percentage of data used for training before specified date
    for group in df.group_by(pl.col("id")):
        b_df = group[1]  # Extract grouped DataFrame
        b_df = b_df.with_row_index("b_idx")
        train_val_b_df = b_df.filter(pl.col("datetime") < split_date)
        test_b_df = b_df.filter(pl.col("datetime") > split_date)

        split_idx = int(len(train_val_b_df) * train_per)
        split_idx_two = test_b_df["b_idx"].first()

        min_len = ds.config["n_in"] + ds.config["n_out"]  # length needed for constructing one pair
        if len(train_val_b_df) != 0:  # might be zero if everything after date
            train_b_df = train_val_b_df.filter(pl.col("b_idx") <= split_idx)
            if remove_per != 0:
                train_b_df = train_b_df.filter(pl.col("b_idx") < int(len(train_b_df) * remove_per)).drop(["b_idx"])
            else:
                train_b_df = train_b_df.drop(["b_idx"])
            if split_idx_two is not None:
                val_b_df = train_val_b_df.filter((pl.col("b_idx") > split_idx).and_(pl.col("b_idx") < split_idx_two)).drop(["b_idx"])
            else:
                # there is no data after the specified date, therefore no test data
                val_b_df = train_val_b_df.filter(pl.col("b_idx") > split_idx).drop(["b_idx"])
            if len(train_b_df) < min_len or len(val_b_df) < min_len:  # if not one example can be produced from train series
                # logger.info(f"Removing series of length {len(b_df)} for ID {group[0]}")
                discarded_ids.append(group[0])
                continue  # if series is too short, discard

            train_dfs.append(train_b_df)
            val_dfs.append(val_b_df)

            ds.train_idxs.extend(train_b_df["index"].to_list())
            ds.val_idxs.extend(val_b_df["index"].to_list())

        if split_idx_two is not None:
            test_b_df = test_b_df.drop(["b_idx"])
            if len(test_b_df) <= min_len:
                discarded_ids.append(group[0])
                continue  # if series is too short, discard

            test_dfs.append(test_b_df)
            ds.test_idxs.extend(test_b_df["index"].to_list())

        else:
            continue  # series not after specified date


    logger.info(f"Removed {len(discarded_ids)} series because they were too short")
    logger.info(f"Remaining series: {len(train_dfs)}")

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
    match config["train_test_split_method"]:
        case "group":
            train_data, val_data, test_data = train_test_split_group_based(ds, train_per)
        case "time":
            train_data, val_data, test_data = train_test_split_time_based(ds, train_per, remove_per=config.get("remove_per", 0.0))
        case "date":
            train_data, val_data, test_data = train_test_split_date_based(ds, datetime.date(2024, 1, 30), remove_per=config["remove_per"])

    # transform to pandas DataFrame input
    target_vars = ["diff"]
    if config["n_out"] > 1:
        target_vars += [f"diff(t+{i})" for i in range(1, config["n_out"])]

    train_data = train_data.sort([pl.col("id"), pl.col("datetime")])
    test_data = test_data.sort([pl.col("id"), pl.col("datetime")])
    val_data = val_data.sort([pl.col("id"), pl.col("datetime")])
    # selects only features and target variable
    ds.X_train = train_data.to_pandas()[list(set(config["features"] + ds.get_noise_feature_names()) - set(target_vars))]
    ds.y_train = train_data.to_pandas()[target_vars]
    ds.X_val = val_data.to_pandas()[list(set(config["features"] + ds.get_noise_feature_names()) - set(target_vars))]
    ds.y_val = val_data.to_pandas()[target_vars]
    ds.X_test = test_data.to_pandas()[list(set(config["features"] + ds.get_noise_feature_names()) - set(target_vars))]
    ds.y_test = test_data.to_pandas()[target_vars]

    # plot_means(X_train, y_train, X_val, y_val, X_test, y_test)
    # plot_std(X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info(f"Train data shape: {ds.X_train.shape}")  # TODO: fix shape output for one feature "diff"
    logger.info(f"Test data shape: {ds.X_test.shape}")
    logger.info(f"Validation data shape: {ds.X_val.shape}")
    logger.info(f"Training on {len(config['features'])} features")
    return ds
