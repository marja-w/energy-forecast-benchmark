import numpy as np
import pandas as pd
import re

import polars as pl
from polars.polars import PanicException
from sympy.codegen.ast import continue_

from src.energy_forecast.config import STATIC_COVARIATES


## DATA PROCESSING ##
## PROCESS DATA FOR LSTM TRAINING ##
def series_to_supervised(df: pl.DataFrame, n_in: int, n_out: int, lag_in: int, lag_out: int, dropnan: bool = True) -> pl.DataFrame:
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
    col_to_drop = ["id"]
    if "datetime" in list(df.columns):
        datetime_column = df["datetime"]
        col_to_drop += ["datetime"]
    else:
        pass
    df = df.drop(col_to_drop).to_pandas()
    # if static_covs_extra:
    #     static_covariates_ = ["datetime"] + list(set(df.columns).intersection(STATIC_COVARIATES))
    #     df_static = df[static_covariates_]  # save for later
    #     df = df.drop(static_covariates_)  # remove for time series creation  # TODO
    c_names = list(df.columns)

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(lag_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"{c_names[j]}(t-{i})" for j in range(len(c_names))]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, lag_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f"{c_names[j]}" for j in range(len(c_names))]
        else:
            names += [f"{c_names[j]}(t+{i})" for j in range(len(c_names))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if "datetime" in col_to_drop:
        datetime_df = pd.DataFrame({"datetime": datetime_column.to_pandas()})
        agg = pd.concat([agg, datetime_df], axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    df = pl.DataFrame(agg)
    # after dropping nulls for lag values use actual n_in, n_out values for features
    # if n_in < lag_in:
    #     for c_name in c_names:
    #         df = df.drop([f"{c_name}(t-{i})" for i in range(lag_in, n_in, -1)])
    # if n_out < lag_out:
    #     for c_name in c_names:
    #         df = df.drop([f"{c_name}(t+{i})" for i in range(n_out, lag_out)])
    df = df.with_columns(pl.lit(b_id).alias("id"))
    return df