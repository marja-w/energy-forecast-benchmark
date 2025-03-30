from datetime import date, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd
import polars as pl
from keras.utils import timeseries_dataset_from_array
from loguru import logger
from numpy import ndarray

from src.energy_forecast.config import REPORTS_DIR


def add_env_energy(df: pl.DataFrame, df_raw: pl.DataFrame, address: str, col_gas: str, col_env: str):
    val_col_env = df_raw.filter((pl.col("adresse") == address) & (pl.col("Title") == col_env))["Val"]
    df_adresse = df_raw.filter((pl.col("adresse") == address) & (pl.col("Title") == col_gas))

    # remove the whole address
    df = df.filter(~(pl.col("adresse") == address))

    # add the values of column 2 to column 1 and drop column 2
    df_adresse = df_adresse.with_columns(val_col_env.alias("val_2")).with_columns(
        pl.col("Val") + pl.col("val_2").alias("Val")).drop(["val_2"])

    # rename title
    df_adresse = df_adresse.with_columns(pl.lit("Energie Gesamt").alias("Title"))
    logger.info(f"Summing {col_gas} and {col_env} for {address}")
    logger.info(f"Length of data is {len(df_adresse)}")

    df = pl.concat([df, df_adresse])
    return df


def get_id_from_address(meta_data: dict, adress: str) -> str:
    for key, value in meta_data.items():
        if value["address"]["street_address"] == adress:
            return key


def remove_leading_zeros(df):
    non_zero_idx = df.with_row_index().filter(pl.col("sum_kwh_diff") > 0)[0]["index"][
        0]  # get index of first row that isnt 0
    df = df.with_row_index().filter(pl.col("index") >= non_zero_idx).drop("index")  # remove first 0s
    df = df.filter(pl.col("sum_kwh") != pl.col("sum_kwh_diff"))  # remove first wrong diff value
    return df


def sum_columns(df, address, col_1, col_2):
    val_col_2 = df.filter((pl.col("adresse") == address) & (pl.col("Title") == col_2))["Val"]
    df_adresse = df.filter((pl.col("adresse") == address) & (pl.col("Title") == col_1))

    # remove the whole address
    df = df.filter(~(pl.col("adresse") == address))

    # add the values of column 2 to column 1 and drop column 2
    df_adresse = df_adresse.with_columns(val_col_2.alias("val_2")).with_columns(
        pl.col("Val") + pl.col("val_2").alias("Val")).drop(["val_2"])

    # rename title to Gaszähler Gesamt
    df_adresse = df_adresse.with_columns(pl.lit("Gaszähler Gesamt").alias("Title"))
    logger.info(f"Summing {col_1} and {col_2} for {address}")
    logger.info(f"Length of data is {len(df_adresse)}")

    df = pl.concat([df, df_adresse])
    return df


def get_missing_dates(df: pl.DataFrame, frequency: str = "D") -> pl.DataFrame:
    """
    Get missing dates/hours, depending on frequency parameter
    :param df: DataFrame with datetime column
    :param frequency: "D" or "h"
    :return: DataFrame with missing dates as list, length of data versus length of missing dates, and start and end date
            of data series
    """
    missing_dates = []
    df_time = df.group_by(pl.col("id")).agg(pl.len(),
                                            pl.col("datetime").min().alias("min_date"),
                                            pl.col("datetime").max().alias("max_date")
                                            )
    for row in df_time.iter_rows():
        id = row[0]
        # print("\nSensor: ", id, "\n")
        # print(f"Sensor {id} has {row[1]} datapoints")
        start_date = row[2]
        end_date = row[3]

        date_list_rec = df.filter(pl.col("id") == id).select(pl.col("datetime"))["datetime"].to_list()

        date_list = pd.date_range(start_date, end_date, freq=frequency)

        missing_dates_sensor = list(set(date_list) - set(date_list_rec))
        missing_dates_sensor.sort()
        # logger.info(f"Missing dates for sensor {id}: {len(missing_dates_sensor)}")
        missing_dates.append(
            {"id": id, "missing_dates": missing_dates_sensor, "len": len(missing_dates_sensor), "n": row[1],
             "per": ((len(date_list_rec) / (1 + len(date_list))) * 100), "start_date": start_date,
             "end_date": end_date})
    df_missing_dates = pl.DataFrame(missing_dates).sort(pl.col("len"), descending=True)
    # write to csv
    missing_dates_csv_ = REPORTS_DIR / "missing_dates.csv"
    df_missing_dates.select(["id", "len", "n", "per", "start_date", "end_date"]).sort(pl.col("per"),
                                                                                      descending=True).write_csv(
        missing_dates_csv_)
    logger.info(f"Wrote information about missing dates to {missing_dates_csv_}")
    return df_missing_dates


def find_time_spans(dates: list[date], delta: timedelta) -> pl.DataFrame:
    if len(dates) == 0:
        return pl.DataFrame()
    spans = []
    try:
        start_date = dates[0]
    except IndexError:
        return pl.DataFrame()
    prev_date = start_date
    count = 1

    for i in range(1, len(dates)):
        current_date = dates[i]

        if current_date == prev_date + delta:
            count += 1
        else:
            spans.append({"start": start_date, "end": prev_date, "n": count})
            start_date = current_date
            count = 1

        prev_date = current_date

    spans.append({"start": start_date, "end": prev_date, "n": count})  # Add the last span
    df_spans = pl.DataFrame(spans)
    return df_spans


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

## OLD COMPANY PROJECT METHODS ##

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

    df = df.group_by("id").map_groups(lambda group: add_last_n_diffs(group, n))
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
