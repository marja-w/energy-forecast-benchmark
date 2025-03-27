import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.energy_forecast.config import PADDING_VALUE, PROJ_ROOT


def get_split(df: pl.DataFrame) -> tuple:
    # Ensure new_id is of type str
    df = df.with_columns(pl.col("new_id").cast(pl.Utf8))

    with open(f"{PROJ_ROOT}/references/train_test_val_split.json", "r", encoding="utf-8") as f:
        split_details = json.load(f)

    train_ids = split_details["train_ids"]
    test_ids = split_details["test_ids"]
    val_ids = split_details["val_ids"]

    train_df = df.filter(pl.col("new_id").is_in(train_ids))
    test_df = df.filter(pl.col("new_id").is_in(test_ids))
    val_df = df.filter(pl.col("new_id").is_in(val_ids))
    return train_df, test_df, val_df


def get_split_with_time(df: pl.DataFrame) -> tuple:
    # Ensure date column is of type Date
    df = df.with_columns(pl.col("date").cast(pl.Date))

    # Define split dates
    train_end_date = datetime(2021, 8, 1).date()
    val_end_date = datetime(2021, 11, 1).date()

    # Split the DataFrame
    train_df = df.filter(pl.col("date") < train_end_date)
    val_df = df.filter((pl.col("date") >= train_end_date) & (pl.col("date") < val_end_date))
    test_df = df.filter(pl.col("date") >= val_end_date)

    return train_df, test_df, val_df


def filter_for_relevant_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter the DataFrame for relevant features
    """
    rel_cols = [
        "new_id", "date", "GMonth", "qmbehfl", "diff", "dayofweek", "weekend", "season",
        "Temp_avg", "Humidity_avg", "is_holiday", "weekday_sin", "weekday_cos"
    ]
    return df[rel_cols]


def create_lag_features(
        df: pd.DataFrame,
        n: int = 3,
        group_col: str = 'new_id',
        lag_col: str = 'diff'
) -> pd.DataFrame:
    """
    Create lag features for a specified number of previous values.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - n: int, default=3
        Number of lag features to create.
    - group_col: str, default='new_id'
        The column to group by (e.g., 'new_id').
    - sort_col: str, default='date'
        The column to sort by within each group (e.g., 'date').
    - lag_col: str, default='diff'
        The column for which to create lag features.

    Returns:
    - pd.DataFrame
        The DataFrame with added lag features.

    Raises:
    - ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("Number of lags 'n' must be at least 1.")

    # Sort the DataFrame to ensure correct lagging
    # df = df.sort_values(by=[group_col, sort_col])

    for lag in range(1, n + 1):
        df[f'{lag}_lag_{lag_col}'] = df.groupby(group_col)[lag_col].shift(lag)

    return df


def create_lag_features_polars(df: pl.DataFrame, lag_col: str, n: int) -> pl.DataFrame:
    for lag in range(1, n + 1):
        df = df.with_columns(
            pl.col(lag_col)
            .shift(lag)
            .over("new_id")  # Gruppierung nach 'new_id'
            .alias(f'{lag_col}_lag_{lag}')
        )
    return df


def preprocess_fcn(df: pl.DataFrame, n_past: int = 7, n_out: int = 2):
    # Selection of Relevant Columns
    df = filter_for_relevant_features(df)
    df = df.filter(~pl.col("Humidity_avg").is_null())
    df = df.filter(pl.col("qmbehfl") > 0)

    # Erstellen von Lag-Features
    df = create_lag_features_polars(df, lag_col='diff', n=n_past)
    df = create_lag_features_polars(df, lag_col='Temp_avg', n=n_past)

    # Zielspalten und vergangene Temperaturspalten (optional, falls benötigt)
    past_diff_columns = [f'diff_lag_{lag}' for lag in range(1, n_past + 1)]
    past_temp_columns = [f'Temp_avg_lag_{lag}' for lag in range(1, n_past + 1)]

    # Zukunftsspalten
    future_columns_num = ["Temp_avg", "Humidity_avg"]
    future_columns_bin = ["weekend", "is_holiday"]

    y_columns = ["diff"]
    for i in range(1, n_out):
        df = df.with_columns(
            pl.col("diff")
            .shift(-i)
            .over("new_id")  # Gruppierung nach 'new_id'
            .alias(f'diff_t+{i}')
        )
        y_columns.append(f'diff_t+{i}')

    # Erstellen der zukünftigen numerischen Spalten
    for future_col in future_columns_num.copy():
        for i in range(1, n_out):
            df = df.with_columns(
                pl.col(future_col)
                .shift(-i)
                .over("new_id")  # Gruppierung nach 'new_id'
                .alias(f'{future_col}_t+{i}')
            )
            future_columns_num.append(f'{future_col}_t+{i}')

    # Erstellen der zukünftigen binären Spalten
    for future_col in future_columns_bin.copy():
        for i in range(1, n_out):
            df = df.with_columns(
                pl.col(future_col)
                .shift(-i)
                .over("new_id")  # Gruppierung nach 'new_id'
                .alias(f'{future_col}_t+{i}')
            )
            future_columns_bin.append(f'{future_col}_t+{i}')

    # Split the dataset into train, test, and validation sets
    train_df, test_df, val_df = get_split(df)

    # Convert to Pandas DataFrames for easier manipulation
    train_df = train_df.to_pandas()
    val_df = val_df.to_pandas()
    test_df = test_df.to_pandas()

    # Optionally, you can drop rows with NaN values resulting from lagging
    # This step depends on whether you want to keep all data or not
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    # Log the number of rows after creating lag features
    logger.info(f"Train DataFrame length after lagging: {len(train_df)}")
    logger.info(f"Validation DataFrame length after lagging: {len(val_df)}")
    logger.info(f"Test DataFrame length after lagging: {len(test_df)}")

    # Define feature columns
    binary_features = ["weekday_sin", "weekday_cos"] + future_columns_bin
    num_features = ["qmbehfl"] + future_columns_num + past_diff_columns + past_temp_columns

    cat_features = ["GMonth", "season", "dayofweek"]
    sensor_index = {"train": train_df["new_id"], "val": val_df["new_id"], "test": test_df["new_id"]}
    date_index = {"train": train_df["date"], "val": val_df["date"], "test": test_df["date"]}

    # Define the preprocessing steps using ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('bin', 'passthrough', binary_features)
        ]
    )
    preprocessor.set_output(transform="pandas")  # keep the column names

    # Fit the preprocessor on the training data
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df[y_columns].values

    # Scale the target variable
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Transform the validation data
    X_val = preprocessor.transform(val_df)
    y_val = val_df[y_columns].values
    y_val_scaled = scaler_y.transform(y_val)

    # Transform the test data
    X_test = preprocessor.transform(test_df)
    y_test = test_df[y_columns].values
    y_test_scaled = scaler_y.transform(y_test)

    preprocessed_data = {
        "train": {
            "X": X_train,
            "y": y_train_scaled
        },
        "val": {
            "X": X_val,
            "y": y_val_scaled
        },
        "test": {
            "X": X_test,
            "y": y_test_scaled
        },
        "scaler_y": scaler_y,
        "preprocessor": preprocessor,
        "sensor_index": sensor_index,
        "date_index": date_index,
        "test_df": test_df
    }

    return preprocessed_data


def input_pipeline(scaler_X, num_features, cat_features):
    # Define the preprocessing steps using ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', scaler_X)
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'
    )
    preprocessor.set_output(transform="pandas")  # keep the column names


def preprocess(df: pl.DataFrame, n: int = 7, lstm: bool = False):
    """
    Preprocess the DataFrame by filtering relevant features, handling missing values,
    creating lag features (if not for LSTM), splitting into train/test/val, and applying transformations.

    Parameters:
    - df: pl.DataFrame
        The input DataFrame.
    - n: int, default=1
        Number of previous y values to include as features.
    - lstm: bool, default=False
        If True, preprocess data for training an LSTM model (lag features are not computed).

    Returns:
    - dict: A dictionary containing preprocessed data:
        - train: dict
            - X: np.ndarray
                Transformed feature matrix for the training set.
            - y: np.ndarray
                Scaled target variable for the training set.
        - val: dict
            - X: np.ndarray
                Transformed feature matrix for the validation set.
            - y: np.ndarray
                Scaled target variable for the validation set.
        - test: dict
            - X: np.ndarray
                Transformed feature matrix for the test set.
            - y: np.ndarray
                Scaled target variable for the test set.
    - scaler_y: StandardScaler
        The scaler fitted on the target variable.
        - sensor_index: dict
            Indices of sensors for train, val, and test sets.
    - test_df: pd.DataFrame
        The test DataFrame in pandas format.
    """

    # Selection of Relevant Columns
    df = filter_for_relevant_features(df)
    df = df.filter(~pl.col("Humidity_avg").is_null())
    df = df.filter(pl.col("qmbehfl") > 0)

    # Split the dataset into train, test, and validation sets
    train_df, test_df, val_df = get_split(df)

    # Convert to Pandas DataFrames for easier manipulation
    train_df = train_df.to_pandas()
    val_df = val_df.to_pandas()
    test_df = test_df.to_pandas()

    if not lstm:
        # Create lag features for train, val, and test sets
        train_df = create_lag_features(train_df, n=n)
        val_df = create_lag_features(val_df, n=n)
        test_df = create_lag_features(test_df, n=n)

    # Optionally, you can drop rows with NaN values resulting from lagging
    # This step depends on whether you want to keep all data or not
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    # Log the number of rows after creating lag features
    logger.info(f"Train DataFrame length after lagging: {len(train_df)}")
    logger.info(f"Validation DataFrame length after lagging: {len(val_df)}")
    logger.info(f"Test DataFrame length after lagging: {len(test_df)}")

    # Define feature columns
    binary_features = ["weekend", "is_holiday", "weekday_sin", "weekday_cos"]
    num_features = ["qmbehfl", "Temp_avg", "Humidity_avg"] + [f'diff_lag_{lag}' for lag in
                                                              range(1, n + 1)] if not lstm else ["qmbehfl", "Temp_avg",
                                                                                                 "Humidity_avg"]
    cat_features = ["GMonth", "season"]
    sensor_index = {"train": train_df["new_id"], "val": val_df["new_id"], "test": test_df["new_id"]}
    date_index = {"train": train_df["date"], "val": val_df["date"], "test": test_df["date"]}

    # Define the preprocessing steps using ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('bin', 'passthrough', binary_features)
        ]
    )
    preprocessor.set_output(transform="pandas")  # keep the column names

    # Fit the preprocessor on the training data
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["diff"].values

    # Scale the target variable
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Transform the validation data
    X_val = preprocessor.transform(val_df)
    y_val = val_df["diff"].values
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Transform the test data
    X_test = preprocessor.transform(test_df)
    y_test = test_df["diff"].values
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    preprocessed_data = {
        "train": {
            "X": X_train,
            "y": y_train_scaled
        },
        "val": {
            "X": X_val,
            "y": y_val_scaled
        },
        "test": {
            "X": X_test,
            "y": y_test_scaled
        },
        "scaler_y": scaler_y,
        "preprocessor": preprocessor,
        "sensor_index": sensor_index,
        "date_index": date_index,
        "test_df": test_df
    }

    return preprocessed_data


## DATA PROCESSING ##
## PROCESS DATA FOR LSTM TRAINING ##

def series_to_supervised(df, config, past_diffs: bool, dropnan=True):
    """
    Frames a multivariate time series as a supervised learning dataset.

    :param config: dictionary containing number of input steps (n_steps_in), number of output steps (n_steps_out),
                    number of future time steps to include in the training data,
                    starting with timestep=t (n_steps_future)
    :param df: Multivariate time series data. It should include the dates in a "date" column. After the dates column,
                the first column is the difference value (target value), the rest are the features to consider.
    :param past_diffs: Whether the past differences are included in training data
    :param dropnan: Whether to drop NaN values after shifting.
    :return: Tuple of input and output arrays for supervised learning, as well as the dates for each datapoint.
    """
    n_in = config["n_steps_in"]
    n_out = config["n_steps_out"]
    n_steps_future = config["n_steps_future"]

    dates = df.pop("date")  # remove date column from data and save dates for later
    n_vars = df.shape[1]
    n_features = n_vars if past_diffs else n_vars - 1  # number of features without the diff values prepended

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j + 1}(t-{i})' for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j + 1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j + 1}(t+{i})' for j in range(n_vars)]

    # put it all together
    agg: pd.DataFrame = pd.concat(cols, axis=1)
    agg = agg.set_axis(names, axis=1)  # rename column names

    # drop rows with NaN values
    if dropnan:
        agg.insert(0, "date", dates)  # add dates again
        agg.dropna(inplace=True)
        dates = agg.pop("date")  # remove again

    y_output = ["var1(t)"] if (n_out == 1) else ["var1(t)"] + [f"var1(t+{x + 1})" for x in range(n_out - 1)]
    y = np.array(agg[y_output]).reshape(-1, n_out)  # get target variable for current timestep
    if n_steps_future == 0:
        agg.drop(columns=agg.columns[-(n_vars * n_out):], axis=1,
                 inplace=True)  # remove values for timestep=t and later
    else:
        agg[y_output] = PADDING_VALUE  # replace the target variable with padding value
        if n_out > n_steps_future:
            agg.drop(columns=agg.columns[-(n_vars * (n_out - n_steps_future)):], axis=1,
                     inplace=True)  # remove values for timesteps greater than n_steps_future
    if not past_diffs:
        # remove diff values from training set
        p = re.compile("var1")  # any var1 variable, var1 is the diff value
        idxs: list[int] = [index for index, string in enumerate(agg.columns) if p.search(string)]
        idxs_: list[str] = list(np.array(agg.columns)[idxs])
        agg.drop(columns=idxs_, axis=1, inplace=True)  # drop every 'var1' col

    X = np.array(agg)
    # reshape into form [samples, timesteps, features]
    X = X.reshape((X.shape[0], n_in + n_steps_future, n_features))  # defined features plus diff
    return X, y, np.array(dates)


def sensors_to_supervised(data, config, idx, dates, past_diffs=True):
    """
    Convert sensor data into a supervised learning dataset.

    :param dates:
    :param n_steps_future:
    :param past_diffs:
    :param data: The sensor data.
    :param n_input: Number of input time steps.
    :param n_out: Number of output time steps.
    :param idx: Determines which datapoints belong to one sensor.
    :return: Tuple of input and output arrays for supervised learning.
    """
    # create dataframe with id as index and add the date column
    df = pd.DataFrame(data, index=idx)
    df.insert(0, "date", list(dates))

    groups = df.groupby("new_id").apply(lambda x: series_to_supervised(x, config, past_diffs))

    X = [x[0] for x in groups]
    y = [x[1] for x in groups]
    dates = [x[2].reshape(-1, 1) for x in groups]
    id_to_length: dict[str, int] = {groups.index[x]: groups[x][0].shape[0] for x in
                                    range(len(groups))}  # number of entries for each sensor
    return np.vstack(X), np.vstack(y), np.vstack(dates), id_to_length
