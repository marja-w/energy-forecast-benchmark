import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import typer
from loguru import logger
from meteostat import Daily, Hourly

from src.energy_forecast.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REFERENCES_DIR
from src.energy_forecast.data_processing.transform import update_df_with_corrections
from src.energy_forecast.util import get_season, remove_vals_by_title, subtract_diff_data, replace_title_values

app = typer.Typer()


def remove_neg_diff_vals(df):
    """
    Remove faulty gas meter data points that caused negative diff values
    :param df:
    :return:
    """

    # die diffs stimmen jetzt nicht mehr, wenn reihen entfernt werden. Problem?
    # die diffs sollten weiterhin stimmen, da die differenz vom "falschen" Gaszählerstand immer noch die richtige
    # differenz ist

    return df.filter(
        pl.col("diff") >= 0  # remove all rows with negative usage
    )


def filter_outliers_iqr(df, column):
    """
    Filter outliers in the specified column of the DataFrame using the 1.5 IQR method.

    :param df: polars DataFrame
    :param column: column name to filter outliers
    :return: DataFrame with outliers removed
    """
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25

    upper_bound = q75 + 1.5 * iqr

    filtered_df = df.filter(pl.col(column) <= upper_bound)
    filtered_count = len(df) - len(filtered_df)

    logger.info(f"Filtered {filtered_count} rows for column {column} for ID {df['new_id'][0]}")

    return filtered_df


def filter_outliers_by_id(df, filter_column):
    """
    Apply the filter_outliers_iqr function to each subset of the DataFrame grouped by the id_column.

    :param df: polars DataFrame
    :param filter_column: column name to filter outliers
    :return: DataFrame with outliers removed for each group
    """

    filtered_df = df.group_by("new_id").map_groups(lambda group: filter_outliers_iqr(group, filter_column))
    return filtered_df


def update_legacy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform legacy data to usable format
    :param df:
    :return:
    """
    df = df.drop(columns=['lastAggregated'])

    # keep only gas values
    df.rename({"CircuitPoint": "primary_energy"}, axis=1, inplace=True)
    df = df[df["primary_energy"] == "GAS"]  # filter for gas values

    # create date column
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'])
    df["date"] = df["ArrivalTime"].dt.date

    # create ID column
    df = df.astype({"GSM_ID": "string"})

    def create_id(row):
        return str(row["GSM_ID"] + row["TP"][:1] + row["Tag"])

    df["new_id"] = df.apply(create_id, axis=1)

    # extract metadata
    df_p = pl.from_pandas(df)
    df_p = df_p.group_by(pl.col("new_id")).agg(pl.col("GSM_ID").max(), pl.col("TP").max(), pl.col("Tag").max(),
                                               pl.col("Title").max(), pl.col("Type").max(), pl.col("s").max(),
                                               pl.col("m").max(), pl.col("CircuitType").max(),
                                               pl.col("CircuitNum").max(), pl.col("co2koeffizient").max(),
                                               pl.col("Objekttitel").max())
    df_p.write_csv(PROCESSED_DATA_DIR / "legacy_meta_data.csv")

    # remove duplicates
    df.set_index(["new_id", "ArrivalTime"], inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # compute diff column
    df['diff'] = df.groupby(['new_id'])['Val'].diff()  # compute difference TODO: Datenlücken bei date? check
    df = df.dropna(subset=["diff"])
    df = df.reset_index()

    ## transform
    df = pl.from_pandas(df)
    # subtract data from Schenfelder Holt 135 BHKW from Gesamt and replace the values
    df = subtract_diff_data(df, "400768GVG", "400768GVA")

    # remove BHKWs and GAPWs from dataset
    df = remove_vals_by_title(df, ["Gaszähler BHKW", "Gaszähler GAWP"])

    # unify naming of gas meters
    df = replace_title_values(df, [("Gas Zähler", "Gaszähler"),
                                   ("Gaszähler Z Kessel", "Gaszähler Kessel Z"),
                                   ("Gas", "Gaszähler"),
                                   ("Gesamt Gaszähler", "Gaszähler Gesamt")])
    logger.info(f"Current length of data: {len(df)} after removing BHKWs and GAPWs")

    # add missing qmbehfl and anzlwhg values from correction file
    correction_csv_path = REFERENCES_DIR / "liegenschaften_missing_qm_wohnung.csv"
    df = update_df_with_corrections(df, correction_csv_path)
    logger.info(f"Current length of data: {len(df)} after adding missing qmbehfl and anzlwhg values")

    # manually remove corrupt days for Wilhelmstraße 33-41
    df = df.filter(~((pl.col("new_id") == "400305GVG") & (
        pl.col("date").is_between(datetime(2019, 11, 6), datetime(2019, 12, 21)))))  # remove corrupt days
    # manually remove corrupt days for Kaltenbergen 22
    df = df.filter(~((pl.col("new_id") == "400204GVA") & (
        pl.col("date").is_between(datetime(2020, 3, 12), datetime(2020, 4, 28)))))
    # manually remove corrupt days for Dahlgrünring 5-9
    df = df.filter(~((pl.col("new_id") == "400711GVG") & (
        pl.col("date").is_between(datetime(2021, 4, 8), datetime(2021, 6, 28)))))

    print(f"Current length of data: {len(df)}")
    logger.success("Processing legacy dataset complete.")

    # convert back to pandas dataframe
    df = df.to_pandas()
    df.drop(
        columns=["GSM_ID", "TP", "Tag", "Title", "CircuitType", "CircuitNum", "Objekttitel", "GHour", "Type", "s", "m",
                 "co2koeffizient"], inplace=True)
    df.set_index(["new_id", "ArrivalTime"], inplace=True)
    return df


def update_kinergy_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename({"hash": "new_id", "unit_code": "Unit"}, axis=1, inplace=True)
    # extract metadata
    df_p = pl.from_pandas(df)
    df_p = df_p.group_by(pl.col("new_id")).agg(pl.col("renewable_energy_used").max(), pl.col("has_pwh").max(),
                                               pl.col("pwh_type").max(),
                                               pl.col("building_type").max(), pl.col("orga").max(),
                                               pl.col("complexity").max(),
                                               pl.col("complexity_score").max(), pl.col("env_id").max())
    df_p.write_csv(PROCESSED_DATA_DIR / "kinergy_meta_data.csv")
    df.drop(columns=["renewable_energy_used", "has_pwh", "pwh_type", "building_type", "orga", "complexity",
                     "complexity_score", "env_id", "avg_sum_kwh", "total_kwh_diff"],
            inplace=True)  # drop columns that arent needed
    df.set_index(["new_id", "ArrivalTime"], inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df


def update_district_heating_data(df: pd.DataFrame) -> pd.DataFrame:
    # district heating data has no attribute hash
    # diff already computed
    df["GYear"] = df["ArrivalTime"].dt.year
    df["GMonth"] = df["ArrivalTime"].dt.month
    df["GDay"] = df["ArrivalTime"].dt.day
    df.set_index(["eco_u_id", "data_provider_id", "ArrivalTime"], inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_data(input_path: Path, data_source: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'])
    if data_source == "legacy":
        df = update_legacy_data(df)
    elif data_source == "kinergy":
        df = update_kinergy_data(df)
    elif data_source == "district_heating":
        df = update_district_heating_data(df)

    df['dayofweek'] = df.index.get_level_values('ArrivalTime').dayofweek
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['season'] = df.apply(lambda row: get_season(row['GMonth'], row['GDay']), axis=1)
    df["source"] = data_source

    df = df.reset_index()
    df.drop("ArrivalTime", axis=1, inplace=True)

    # outlier detection and removal of erroneous values
    df = pl.from_pandas(df)
    df = remove_neg_diff_vals(df)
    logger.info(f"Current length of data: {len(df)} after removing negative diffs")

    df = filter_outliers_by_id(df, "diff")
    logger.info(f"Current length of data: {len(df)} after filtering outliers")
    print(f"Current length of data: {len(df)}")
    df = df.to_pandas()

    store_csv = PROCESSED_DATA_DIR / f"{data_source}_pre_merge.csv"
    if not os.path.exists(store_csv): df.to_csv(store_csv)
    return df


def load_env_data(input_path_env: Path):
    """

    :param input_path_env: path to file with environment data
    :return: dataframe with environment data
    """
    wetter_df = pd.read_csv(input_path_env)
    wetter_df.rename(columns={"City": "ort"}, inplace=True)
    return wetter_df


def load_env_data_meteostat(start, end, location, location_name):
    data = Daily(location, start, end)
    data = data.fetch()
    data["GYear"] = data.index.year
    data["GMonth"] = data.index.month
    data["GDay"] = data.index.day

    # get hourly data to group for daily humidity vals
    hourly_data = Hourly(location, start, end)
    hourly_data = hourly_data.fetch()
    hourly_data["date"] = hourly_data.index.date
    data["Humidity_min"] = hourly_data.groupby("date").min()["rhum"]
    data["Humidity_max"] = hourly_data.groupby("date").max()["rhum"]
    data["Humidity_avg"] = hourly_data.groupby("date").mean()["rhum"]
    data["ort"] = location_name

    data.rename(columns={"tavg": "Temp_avg", "tmin": "Temp_min", "tmax": "Temp_max"}, inplace=True)
    return data


def merge_datasets(counter_df: pd.DataFrame, env_df: pd.DataFrame, store_csv=None):
    """

    :param counter_df:
    :param env_df:
    :param store: set to true if you want to store the DataFrame as .csv
    :return: dataframe with merged data
    """
    df = counter_df.merge(env_df, on=["ort", "GYear", "GMonth", "GDay"], how="left")
    if "Humidity_min" in df.columns:
        try:
            df['Humidity_min'] = df['Humidity_min'].str.replace("%", "")  # replace the '%' sign added in some rows
            df['Humidity_max'] = df['Humidity_max'].str.replace("%", "")  # replace the '%' sign added in some rows
        except AttributeError:
            pass
    try:
        df['plz'] = df['plz'].str.strip()  # replace empty chars before and after
    except AttributeError:
        pass
    if store_csv: df.to_csv(store_csv)
    return df


def merge_counter_env_data(input_path_counter: Path, input_path_env: Path, data_source: str) -> pd.DataFrame:
    logger.info(f"Loading {data_source} data")
    counter_df = load_data(input_path_counter, data_source=data_source)
    env_df = load_env_data(input_path_env)
    df: pd.DataFrame = merge_datasets(counter_df, env_df)
    logger.info(f"Length of {data_source} data: {len(df)}")
    return df


@app.command()
def main(
        input_path_counter: Path = RAW_DATA_DIR / "legacy_systen_counter_daily_values.csv",
        input_path_env: Path = RAW_DATA_DIR / "legacy_system_environment_daily_values.csv",
        output_path: Path = PROCESSED_DATA_DIR / "dataset_legacy_kinergy.csv"
):
    input_path_meteostat = RAW_DATA_DIR / "meteostat_env_data.csv"
    # legacy data
    legacy_df = merge_counter_env_data(input_path_counter, input_path_env, data_source="legacy")
    # merge with meteostat data to remove null values in temperature
    env_df = load_env_data(input_path_meteostat)
    legacy_df = merge_datasets(legacy_df, env_df)
    for col_name in ["Temp_avg", "Temp_min", "Temp_max", "Humidity_min", "Humidity_max", "Humidity_avg"]:
        legacy_df[col_name] = legacy_df[f"{col_name}_x"].combine(legacy_df[f"{col_name}_y"],
                                                                 lambda x, y: x if np.isnan(y) else y)
        legacy_df.drop(columns=[f"{col_name}_x", f"{col_name}_y"], inplace=True)

    # kinergy data
    input_path_kinergy = RAW_DATA_DIR / "kinergy_daily_values.csv"
    kinergy_df = merge_counter_env_data(input_path_kinergy, input_path_meteostat, data_source="kinergy")
    # substitute all null values with temperature values from meteostat
    kinergy_df["Temp_avg"] = kinergy_df["Temp_avg"].combine(kinergy_df["avg_env_temp"],
                                                            lambda x, y: x if np.isnan(y) else y)
    kinergy_df.drop(columns=["avg_env_temp"], inplace=True)

    # merge all data sources
    df = pd.concat([legacy_df, kinergy_df])
    df["date"] = pd.to_datetime(df["date"])  # bring date column to one format

    # logger.info("Loading district heating data")
    # input_path_dh = RAW_DATA_DIR / "district_heating_daily_merge.csv"
    # counter_df = load_data(input_path_dh, data_source="district_heating")
    # location = Point(53.708314, 9.992318)  # Norderstedt
    # env_df = load_env_data_meteostat(counter_df["ArrivalTime"].min(), counter_df["ArrivalTime"].max(), location,
    #                                  "Norderstedt")
    # store_csv = PROCESSED_DATA_DIR / "consumption-cleaned-dh-merged-weather-meteostat.csv"
    # merge_df = merge_datasets(counter_df, env_df, store_csv)
    # df = pd.concat([df, merge_df])
    # print(f"Current length of data: {len(df)}")

    df.to_csv(output_path)


if __name__ == "__main__":
    app()
