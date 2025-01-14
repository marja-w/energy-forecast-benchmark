from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import json
import os.path

from src.energy_forecast.config import DATA_DIR, RAW_DATA_DIR

orga_to_ort = {"BEM": "Berlin", "WWG": "Würzburg", "SBH": "Hamburg", "Jo-Sti": "Bamberg", "Ploen": "Ploen"}


def consumption_extractor(data: pd.DataFrame, eco_u_data, eco_u_id) -> tuple[Any, float | int | Any]:
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


def consumption_extractor_dh(data: pd.DataFrame, data_sensor, eco_u_id) -> tuple[Any, float | int | Any]:
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


def convert_kinergy_data():
    kinergy_folder = DATA_DIR / "kinergy"
    input_folder = kinergy_folder / "consumption_data"
    output_folder = DATA_DIR / "raw"
    eco_u_data_file = kinergy_folder / "kinergy_eco_u_list.json"

    all_data = pd.DataFrame()
    hours = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        print(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['name']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_consumption.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_consumption.csv")
            consumption_data, delta_hours = consumption_extractor(data=data_raw, eco_u_data=eco_u_data,
                                                                  eco_u_id=eco_u_id)
            print(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            delta = consumption_data['date'].max() - consumption_data['date'].min()
            hours.append(delta_hours)
            print("\n")
        else:
            print(f"Missing file for: {input_folder}/{eco_u_id}.csv")
    all_data.to_csv(output_folder / "kinergy_hourly_values.csv", index=False)
    print(f"Data saved to {output_folder}/kinergy_hourly_values.csv")
    print(f"Total {len(all_data)} datapoints")
    print(f"Average {sum(hours) / len(hours)} hours per sensor")
    print(f"Maximum hours: {max(hours)}")
    print(f"Minimum hours: {min(hours)}")


def convert_dh_data():
    dh_data_folder = RAW_DATA_DIR / "district_heating_data"
    input_folder = dh_data_folder / "data"
    eco_u_data_file = dh_data_folder / "eco_u_ids.json"
    output_folder = DATA_DIR / "raw"
    output_csv = output_folder / "district_heating_hourly.csv"

    all_data = pd.DataFrame()
    hours = list()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())
    for eco_u_id in list(eco_u_data.keys()):
        print(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['foreign_identifier']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_1.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_1.csv")
            consumption_data, delta_hours = consumption_extractor_dh(data=data_raw, data_sensor=eco_u_data[eco_u_id],
                                                                     eco_u_id=eco_u_id)
            print(f"Adding {len(consumption_data)} datapoints")
            all_data = pd.concat([all_data, consumption_data])
            delta = consumption_data['date'].max() - consumption_data['date'].min()
            hours.append(delta_hours)
            print("\n")
        else:
            print(f"Missing file for: {input_folder}/{eco_u_id}.csv")
    all_data.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
    print(f"Total {len(all_data)} datapoints")
    print(f"Average {sum(hours) / len(hours)} hours per sensor")
    print(f"Maximum hours: {max(hours)}")
    print(f"Minimum hours: {min(hours)}")


if __name__ == '__main__':
    # convert_kinergy_data()
    convert_dh_data()
