import json
import os
from typing import Any

import pandas as pd
import polars as pl
import typer
from loguru import logger

from config import RAW_DATA_DIR, DATA_DIR
from src.energy_forecast.config import REPORTS_DIR

app = typer.Typer()


def consumption_extractor_kinergy_daily(data: pd.DataFrame, eco_u_data, eco_u_id) -> pd.DataFrame:
    # add column datetime
    df = data.with_columns(pl.col("bucket").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("datetime"))

    df = df.select(["datetime", "sum_kwh", "sum_kwh_diff", "env_temp"])  # Remove unnecessary columns

    df = df.filter(pl.col("sum_kwh") > 0,  # remove first 0 rows
                   pl.col("sum_kwh") != pl.col("sum_kwh_diff")  # remove first wrong diff value
                   )

    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date")
    )

    # aggregate values to daily values
    df_daily = df.group_by("date").agg([
        pl.col("env_temp").mean().alias("avg_env_temp"),
        pl.col("sum_kwh").mean().alias("avg_sum_kwh"),
        pl.col("sum_kwh_diff").sum().alias("total_kwh_diff")
    ]).sort(by="date")

    # rename
    df = df_daily.to_pandas()  # make pandas dataframe
    df["ArrivalTime"] = df["date"]
    df["GYear"] = df["date"].dt.year
    df["GMonth"] = df["date"].dt.month
    df["GDay"] = df["date"].dt.day
    df["Val"] = df["avg_sum_kwh"]
    df["diff"] = df["total_kwh_diff"]

    # merge info from json
    df["qmbehfl"] = eco_u_data[eco_u_id]["heated_area"]
    df["adresse"] = eco_u_data[eco_u_id]["name"]
    df["ort"] = eco_u_data[eco_u_id]["ort"]
    df["plz"] = str(eco_u_data[eco_u_id]["plz"])
    df["anzlwhg"] = eco_u_data[eco_u_id]["anzahlwhg"]

    # other info
    df["orga"] = eco_u_data[eco_u_id]["orga"]  # name of company that owned the building
    df["complexity"] = eco_u_data[eco_u_id]["complexity"]  # complexity rang of the heating system
    df["primary_energy"] = eco_u_data[eco_u_id]["primary_energy"]
    df["complexity_score"] = eco_u_data[eco_u_id]["complexity_score"]
    df["unit_code"] = eco_u_data[eco_u_id]["unit_code"]
    df["renewable_energy_used"] = eco_u_data[eco_u_id]["renewable_energy_used"]
    df["env_id"] = eco_u_data[eco_u_id]["env_id"]  # environment sensor for temperature data
    df["hash"] = eco_u_data[eco_u_id]["hash"]
    df["has_pwh"] = eco_u_data[eco_u_id]["has_pwh"]
    df["pwh_type"] = eco_u_data[eco_u_id]["pwh_type"]
    df["building_type"] = eco_u_data[eco_u_id]["typ"]

    return df


def consumption_extractor_kinergy_hourly(data: pd.DataFrame, eco_u_data, eco_u_id) -> pd.DataFrame:
    # add column datetime
    df = data.with_columns(pl.col("bucket").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("datetime"))

    df = df.select(["datetime", "sum_kwh", "sum_kwh_diff", "env_temp"])  # Remove unnecessary columns

    df = df.filter(pl.col("sum_kwh") > 0,  # remove first 0 rows
                   pl.col("sum_kwh") != pl.col("sum_kwh_diff")  # remove first wrong diff value
                   )

    min_datetime = df["datetime"].min()
    max_datetime = df["datetime"].max()
    delta = max_datetime - min_datetime
    delta_hours = (delta.total_seconds() / 3600) + 2
    print(f"Span from {min_datetime} to {max_datetime}: {delta_hours} hours")

    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
    )

    # aggregate values to hourly values
    df_hourly = df.group_by(["date", "hour"]).agg([
        pl.col("env_temp").mean().alias("avg_env_temp"),
        pl.col("sum_kwh").mean().alias("avg_sum_kwh"),
        pl.col("sum_kwh_diff").sum().alias("total_kwh_diff")
    ]).sort(by=["date", "hour"])

    # rename
    df = df_hourly.to_pandas()  # make pandas dataframe

    # merge info from json
    df["qmbehfl"] = eco_u_data[eco_u_id]["heated_area"]
    df["adresse"] = eco_u_data[eco_u_id]["name"]
    df["ort"] = eco_u_data[eco_u_id]["ort"]
    df["plz"] = eco_u_data[eco_u_id]["plz"]
    df["anzlwhg"] = eco_u_data[eco_u_id]["anzahlwhg"]

    # other info
    df["orga"] = eco_u_data[eco_u_id]["orga"]  # name of company that owned the building
    df["complexity"] = eco_u_data[eco_u_id]["complexity"]  # complexity rang of the heating system
    df["primary_energy"] = eco_u_data[eco_u_id]["primary_energy"]
    df["complexity_score"] = eco_u_data[eco_u_id]["complexity_score"]
    df["unit_code"] = eco_u_data[eco_u_id]["unit_code"]
    df["renewable_energy_used"] = eco_u_data[eco_u_id]["renewable_energy_used"]
    df["env_id"] = eco_u_data[eco_u_id]["env_id"]  # environment sensor for temperature data
    df["hash"] = eco_u_data[eco_u_id]["hash"]
    df["has_pwh"] = eco_u_data[eco_u_id]["has_pwh"]
    df["pwh_type"] = eco_u_data[eco_u_id]["pwh_type"]
    df["building_type"] = eco_u_data[eco_u_id]["typ"]

    return df, delta_hours


def consumption_extractor_dh_daily(data: pd.DataFrame, eco_u_data, eco_u_id) -> pd.DataFrame:
    # add column datetime
    df = data.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("datetime"))

    df = df.select(["datetime", "value"])  # Remove unnecessary columns

    # compute diff column
    df = df.with_columns(change=pl.col("value").diff()).drop_nulls()

    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
    )

    # aggregate values to daily values
    df_daily = df.group_by("date").agg([
        pl.col("change").sum().alias("diff")
    ]).sort(by="date")

    # rename
    df = df_daily.to_pandas()  # make pandas dataframe

    # merge info from json
    df["qmbehfl"] = eco_u_data[eco_u_id]["heated_area"]
    df["adresse"] = eco_u_data[eco_u_id]["foreign_identifier"]
    df["ort"] = eco_u_data[eco_u_id]["address"]["address_locality"]
    df["plz"] = eco_u_data[eco_u_id]["address"]["postal_code"]

    id, data_provider_id = eco_u_id.split(".")
    df["eco_u_id"] = id
    df["data_provider_id"] = data_provider_id

    # other info
    df["primary_energy"] = "district heating"
    df["unit_code"] = "kwh"

    return df


def consumption_extractor_dh_hourly(data: pd.DataFrame, data_sensor, eco_u_id) -> tuple[Any, float | int | Any]:
    # add column datetime
    df = data.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("datetime"))

    df = df.select(["datetime", "value"])  # Remove unnecessary columns

    # compute diff column
    df = df.with_columns(change=pl.col("value").diff()).drop_nulls()

    min_datetime = df["datetime"].min()
    max_datetime = df["datetime"].max()
    delta = max_datetime - min_datetime
    delta_hours = (delta.total_seconds() / 3600) + 2
    print(f"Span from {min_datetime} to {max_datetime}: {delta_hours} hours")

    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
    )

    # aggregate values to hourly values
    df_hourly = df.group_by(["date", "hour"]).agg([
        pl.col("change").sum().alias("diff")
    ]).sort(by=["date", "hour"])

    # rename
    df = df_hourly.to_pandas()  # make pandas dataframe

    # merge info from json
    df["qmbehfl"] = data_sensor["heated_area"]
    df["adresse"] = data_sensor["foreign_identifier"]
    df["ort"] = data_sensor["address"]["address_locality"]
    df["plz"] = data_sensor["address"]["postal_code"]

    eco_u_id, data_provider_id = eco_u_id.split(".")
    df["eco_u_id"] = eco_u_id
    df["data_provider_id"] = data_provider_id

    # other info
    df["primary_energy"] = "district heating"
    df["unit_code"] = "kwh"

    return df, delta_hours


def daily_conversion_kinergy(eco_u_data_file, input_folder, kinergy_daily_output_csv):
    all_data = pd.DataFrame()
    days = list()
    info = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        eco_u_name = eco_u_data[eco_u_id]['name']
        logger.info(f"EcoU {eco_u_id} - {eco_u_name}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_consumption.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_consumption.csv")
            consumption_data = consumption_extractor_kinergy_daily(data=data_raw, eco_u_data=eco_u_data,
                                                                   eco_u_id=eco_u_id)
            missing_data_counter = 0  # TODO
            logger.info(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            date_min = consumption_data['date'].min()
            date_max = consumption_data['date'].max()
            delta = date_max - date_min
            delta_days = delta.days + 2  # not inclusive
            logger.info(
                f"Span from {date_min} to {date_max}: {delta_days} days")
            days.append(delta_days)
            info.append(
                {"id": eco_u_id, "source": "kinergy", "energy_type": eco_u_data[eco_u_id]["primary_energy"],
                 "datapoints_daily": len(consumption_data), "address": eco_u_name,
                 "postal_code": eco_u_data[eco_u_id]["plz"],
                 "start_date": date_min, "end_date": date_max, "missing_data": missing_data_counter})
        else:
            logger.info(f"Missing file for: {input_folder}/{eco_u_id}.csv")
    all_data.to_csv(kinergy_daily_output_csv, index=False)
    logger.info(f"Data saved to {kinergy_daily_output_csv}")
    logger.info(f"Total {len(all_data)} datapoints")
    logger.info(f"Average {sum(days) / len(days)} days per sensor")
    logger.info(f"Maximum days: {max(days)}")
    logger.info(f"Minimum days: {min(days)}")

    return pd.DataFrame(info)


def hourly_conversion_kinergy(eco_u_data_file, input_folder, kinergy_hourly_output_csv):
    all_data = pd.DataFrame()
    hours = list()
    info = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        logger.info(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['name']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_consumption.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_consumption.csv")
            consumption_data, delta_hours = consumption_extractor_kinergy_hourly(data=data_raw, eco_u_data=eco_u_data,
                                                                                 eco_u_id=eco_u_id)
            logger.info(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            hours.append(delta_hours)
            info.append(
                {"id": eco_u_id, "datapoints_hourly": len(consumption_data)})
        else:
            logger.info(f"Missing file for: {input_folder}/{eco_u_id}.csv")

    all_data.to_csv(kinergy_hourly_output_csv, index=False)
    logger.info(f"Data saved to {kinergy_hourly_output_csv}")
    logger.info(f"Total {len(all_data)} datapoints")
    logger.info(f"Average {sum(hours) / len(hours)} hours per sensor")
    logger.info(f"Maximum hours: {max(hours)}")
    logger.info(f"Minimum hours: {min(hours)}")

    return pd.DataFrame(info)


def daily_conversion_dh(dh_daily_output_csv, eco_u_data_file, input_folder):
    all_data = pd.DataFrame()
    days = list()
    info = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        identifier_obj = eco_u_data[eco_u_id]['foreign_identifier']
        logger.info(f"EcoU {eco_u_id} - {identifier_obj}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_1.csv") and os.path.isfile(f"{input_folder}/{eco_u_id}_2.csv"):
            # read in first part
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_1.csv")
            consumption_data_1 = consumption_extractor_dh_daily(data=data_raw, eco_u_data=eco_u_data, eco_u_id=eco_u_id)

            # read in second part
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_2.csv")
            consumption_data_2 = consumption_extractor_dh_daily(data=data_raw, eco_u_data=eco_u_data, eco_u_id=eco_u_id)

            # merge data
            consumption_data = pd.concat([consumption_data_1, consumption_data_2])
            missing_data_counter = 0  # TODO

            # add to all data file
            logger.info(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            date_max = consumption_data['date'].max()
            date_min = consumption_data['date'].min()
            delta = date_max - date_min
            delta_days = delta.days + 2  # not inclusive
            logger.info(
                f"Span from {date_min} to {date_max}: {delta_days} days")
            days.append(delta_days)
            info.append(
                {"id": eco_u_id, "source": "dh", "energy_type": "district heating",
                 "datapoints_daily": len(consumption_data), "address": identifier_obj,
                 "postal_code": eco_u_data[eco_u_id]["address"]["postal_code"],
                 "start_date": date_min, "end_date": date_max, "missing_data": missing_data_counter})
            logger.info("\n")
        else:
            logger.info(f"Missing file for: {input_folder}/{eco_u_id}.csv")

    all_data.to_csv(dh_daily_output_csv, index=False)
    logger.info(f"Data saved to {dh_daily_output_csv}")
    logger.info(f"Total {len(all_data)} datapoints")
    logger.info(f"Average {sum(days) / len(days)} days per sensor")
    logger.info(f"Number of sensors: {len(list(eco_u_data.keys()))}")
    logger.info(f"Maximum days: {max(days)}")
    logger.info(f"Minimum days: {min(days)}")

    return pd.DataFrame(info)


def hourly_conversion_dh(dh_hourly_output_csv, eco_u_data_file, input_folder):
    all_data = pd.DataFrame()
    hours = list()
    info = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        logger.info(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['foreign_identifier']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_1.csv") and os.path.isfile(f"{input_folder}/{eco_u_id}_2.csv"):
            # read in first part
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_1.csv")
            consumption_data_1, delta_hours_1 = consumption_extractor_dh_hourly(data=data_raw,
                                                                                data_sensor=eco_u_data[eco_u_id],
                                                                                eco_u_id=eco_u_id)
            # read in second part
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_2.csv")
            consumption_data_2, delta_hours_2 = consumption_extractor_dh_hourly(data=data_raw,
                                                                                data_sensor=eco_u_data[eco_u_id],
                                                                                eco_u_id=eco_u_id)
            # merge data
            consumption_data = pd.concat([consumption_data_1, consumption_data_2])
            delta_hours = delta_hours_1 + delta_hours_2

            # add to all data file
            logger.info(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            hours.append(delta_hours)
            info.append({"id": eco_u_id, "datapoints_hourly": len(consumption_data)})
        else:
            logger.info(f"Missing file for: {input_folder}/{eco_u_id}.csv")

    all_data.to_csv(dh_hourly_output_csv, index=False)
    logger.info(f"Data saved to {dh_hourly_output_csv}")
    logger.info(f"Total {len(all_data)} datapoints")
    logger.info(f"Average {sum(hours) / len(hours)} hours per sensor")
    logger.info(f"Maximum hours: {max(hours)}")
    logger.info(f"Minimum hours: {min(hours)}")

    return pd.DataFrame(info)


@app.command()
def main(
):  ## KINERGY DATA
    kinergy_folder = DATA_DIR / "kinergy"
    input_folder = kinergy_folder / "consumption_data"
    eco_u_data_file = kinergy_folder / "kinergy_eco_u_list.json"

    ## DAILY CONVERSION
    kinergy_daily_output_csv = RAW_DATA_DIR / "kinergy_daily.csv"
    df_info = daily_conversion_kinergy(eco_u_data_file, input_folder, kinergy_daily_output_csv)

    ## HOURLY CONVERSION
    kinergy_hourly_output_csv = RAW_DATA_DIR / "kinergy_hourly.csv"
    df_info_hourly = hourly_conversion_kinergy(eco_u_data_file, input_folder, kinergy_hourly_output_csv)
    df_info = pd.merge(df_info, df_info_hourly, how="left", on="id")

    ## DISTRICT HEATING DATA
    dh_data_folder = DATA_DIR / "district_heating_data"
    input_folder = dh_data_folder / "data"
    eco_u_data_file = dh_data_folder / "eco_u_ids.json"

    ## DAILY CONVERSION
    dh_daily_output_csv = RAW_DATA_DIR / "district_heating_daily.csv"
    df_more_info = daily_conversion_dh(dh_daily_output_csv, eco_u_data_file, input_folder)

    ## HOURLY CONVERSION
    dh_hourly_output_csv = RAW_DATA_DIR / "district_heating_hourly.csv"
    df_more_info_hours = hourly_conversion_dh(dh_hourly_output_csv, eco_u_data_file, input_folder)
    df_more_info = pd.merge(df_more_info, df_more_info_hours, how="left", on="id")

    pd.concat([df_info, df_more_info]).to_csv(REPORTS_DIR / "info_raw_data.csv")


if __name__ == "__main__":
    app()
