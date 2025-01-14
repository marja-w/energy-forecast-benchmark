from typing import Tuple, List

import numpy as np
import pandas as pd
import polars as pl
from keras.utils import timeseries_dataset_from_array
from numpy import ndarray


def get_season(month, day):
    if (month == 3 and day >= 21) or (3 < month < 6) or (month == 6 and day < 21):
        return 'Spring'
    if (month == 6 and day >= 21) or (6 < month < 9) or (month == 9 and day < 23):
        return 'Summer'
    if (month == 9 and day >= 23) or (9 < month < 12) or (month == 12 and day < 21):
        return 'Fall'
    return 'Winter'


def subtract_diff_data(df, gas_id, remove_id):
    """
    Subtract the diff values from the gas meter with ID remove_id, from the gas meter with ID gas_id.
    Both gas meters must have the same number of examples.
    :param df:
    :param gas_id:
    :param remove_id:
    :return:
    """
    assert len(df.filter(pl.col("new_id") == gas_id)) == len(df.filter(pl.col("new_id") == remove_id))
    df_gesamt = df.filter(pl.col("new_id") == gas_id)
    df_gesamt = df_gesamt.with_columns((df.filter(pl.col("new_id") == remove_id)["diff"]).alias("bhkw"))
    df_gesamt = df_gesamt.with_columns((df_gesamt["diff"] - df_gesamt["bhkw"]).alias("wo_bhkw"))
    df_no_bhkw = df.filter(pl.col("new_id") == gas_id).replace_column(df.get_column_index("diff"),
                                                                      df_gesamt["wo_bhkw"].alias("diff")
                                                                      ).replace_column(
        df.get_column_index("Title"),
        pl.Series("Title", ["Gaszähler" for _ in range(len(df.filter(pl.col("new_id") == gas_id)))]))
    # drop rows with sum of gas and bhkw
    df = df.filter(~(pl.col("new_id") == gas_id))
    # add difference to DataFrame
    df = pl.concat([df, df_no_bhkw])
    return df


def remove_vals_by_title(df: pl.DataFrame, titles: List[str]):
    """
    Remove all values correlating to the title in the titles list from the dataset
    :param df: polars DataFrame
    :param titles: list of titles to remove
    :return:
    """
    for title in titles:
        df = df.filter(~(pl.col("Title") == title))
    return df


def replace_title_values(df, titles_replace: List[Tuple[str, str]]):
    """
    Replace title values indicated by the list of tuples
    :param df:
    :param titles_replace: list of tuples, each tuples is of the form (title_to_be_replaced, new_title)
    :return:
    """
    for (title_to_be_replaced, new_title) in titles_replace:
        df = df.with_columns(
            pl.when(pl.col("Title") == title_to_be_replaced).then(pl.lit(new_title)).otherwise("Title").alias("Title"))
    return df


def create_diff(df: pl.DataFrame):
    """
    Create the column diff, which records the difference of gas meter value between two data points
    that follow each other according to the date value
    :param df:
    :return:
    """
    ids = [id[0] for id in df.select(pl.col("new_id").unique()).iter_rows()]  # get list of unique ids
    dfs = []

    for gas_id in ids:  # iterate over every gas meter
        id_df = df.filter(
            pl.col("new_id") == gas_id  # look at one gas meter at a time
        ).sort("date").with_columns(  # make sure the data is sorted by date
            pl.col("Val").diff()
        )
        dfs.append(id_df)

    return pl.concat(dfs)


def add_last_n_diffs(df: pl.DataFrame, n: int) -> pl.DataFrame:
    generator = timeseries_dataset_from_array(df["diff"], df["diff"], sequence_length=n,
                                              sequence_stride=1, sampling_rate=1, batch_size=None)
    inputs: ndarray = np.array([inputs.numpy() for (inputs, targets) in generator])
    df_inputs: pl.DataFrame = pl.from_numpy(data=inputs)
    df_inputs.columns = [f"t-{x + 1}" for x in reversed(range(n))]
    return pl.concat([df.tail(-n), df_inputs], how="horizontal").drop_nulls("diff")


def add_last_n_diffs_all_gas_meters(df, n):
    """
    Add the last n observations of the difference variable to the DataFrame
    :param df: polars DataFrame
    :param n: number of steps to look back to
    :return: polars DataFrame with added columns (t-n, t-(n-1),..., t-1)
    """

    df = df.group_by("new_id").map_groups(lambda group: add_last_n_diffs(group, n))
    return df


def get_date_ranges(df: pl.DataFrame):
    return df.group_by("new_id").agg(
        [pl.col("date").min().alias("min_date"),
         pl.col("date").max().alias("max_date")])


def get_date_range_from_id(df: pl.DataFrame, gas_id: str):
    dates_info = df.group_by("new_id").agg(
        [pl.col("date").min().alias("min_date"),
         pl.col("date").max().alias("max_date")]
    ).filter(pl.col("new_id") == gas_id)
    return pd.date_range(dates_info["min_date"][0], dates_info["max_date"][0], freq="D")


def get_address_from_id(df: pl.DataFrame, gas_id: str):
    return df.filter(pl.col("new_id") == gas_id)["adresse"].unique()[0]


def realign_predictions(predictions, sensor_data_lengths, t_in, t_out):
    """
    Realigns the predictions to match the original sensor data lengths.

    Args:
        predictions (np.ndarray): Array of predictions.
        sensor_data_lengths (list): List of lengths of the sensor data.
        t_in (int): Number of input time steps.
        t_out (int): Number of output time steps.

    Returns:
        list: List of realigned predictions for each sensor.
    """
    # Number of sensors
    n_sensors = len(sensor_data_lengths)

    # Placeholder to store the realigned predictions for each sensor
    realigned_predictions = [np.zeros(sensor_data_lengths[i]) for i in range(n_sensors)]

    # Track the index for the start of each prediction in the sensor data
    current_start_idx = [0] * n_sensors  # Starting index for each sensor

    # Iterate over all the predictions
    prediction_idx = 0
    for sensor_idx in range(n_sensors):
        while True:
            # Get the starting index for this sensor
            start_idx = current_start_idx[sensor_idx]

            # Check if the prediction would exceed the sensor's length
            if start_idx + t_in + t_out > sensor_data_lengths[sensor_idx]:
                # If true, we are done with this sensor
                break

            # Place the prediction into the correct part of the sensor data
            datapoint = predictions[prediction_idx]
            realigned_predictions[sensor_idx][start_idx: start_idx + len(datapoint)] = datapoint

            # Update the prediction index and starting point for the next prediction
            prediction_idx += 1
            current_start_idx[sensor_idx] += 1  # Shift start by one for sliding window

    return realigned_predictions
