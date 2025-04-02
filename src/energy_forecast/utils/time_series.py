import numpy as np
import pandas as pd
import re

import polars as pl

from src.energy_forecast.config import PADDING_VALUE


## DATA PROCESSING ##
## PROCESS DATA FOR LSTM TRAINING ##
def series_to_supervised(df: pl.DataFrame, n_in: int = 1, n_out: int = 1, dropnan: bool = True) -> pl.DataFrame:
    """
    Convert series to supervised learning
    Args:
        data:
        n_in:
        n_out:
        dropnan:

    Returns:

    """
    b_id = df["id"].mode().item()
    df = df.drop("id").to_pandas()
    c_names = list(df.columns)

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"{c_names[j]}(t-{i})" for j in range(len(c_names))]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f"{c_names[j]}(t)" for j in range(len(c_names))]
        else:
            names += [f"{c_names[j]}(t+{i})" for j in range(len(c_names))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    df = pl.DataFrame(agg)
    df = df.with_columns(pl.lit(b_id).alias("id"))
    return df


def series_to_supervised_old(df: pd.DataFrame, config: dict, past_diffs: bool, dropnan=True):
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
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_steps_future = config["n_future"]

    dates = df.pop("datetime")  # remove date column from data and save dates for later
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


def sensors_to_supervised(data: pl.DataFrame, config: dict, past_diffs=True):
    """
    Convert sensor data into a supervised learning dataset.

    :param config:
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
    df = data.to_pandas()

    groups = df.groupby("id").apply(lambda x: series_to_supervised(x, config, past_diffs))

    X = [x[0] for x in groups]
    y = [x[1] for x in groups]
    dates = [x[2].reshape(-1, 1) for x in groups]
    id_to_length: dict[str, int] = {groups.index[x]: groups[x][0].shape[0] for x in
                                    range(len(groups))}  # number of entries for each sensor
    return np.vstack(X), np.vstack(y), np.vstack(dates), id_to_length
