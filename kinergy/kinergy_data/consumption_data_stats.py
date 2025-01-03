from datetime import datetime

import pandas as pd
import polars as pl
import json
import os.path

input_folder = "../kinergy_data/consumption_data"
eco_u_data_file = "../kinergy_data/kinergy_eco_u_list.json"


def consumption_extractor(data: pd.DataFrame):
    df = data.with_columns(
        pl.col("bucket")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ")
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        .alias("datetime")
    )  # add column datetime

    df = df.select(["datetime", "sum_kwh", "sum_kwh_diff", "env_temp"])  # Remove unnecessary columns

    start_date: str = eco_u_data[eco_u_id]["begin"]
    start_datetime = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:%M:%S")

    end_date: str = eco_u_data["1a9266de-dfff-11eb-9d61-02b402f0c1de"]["end"]
    end_datetime = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:%M:%S")

    duration = datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S") - datetime.strptime(start_datetime,
                                                                                        "%Y-%m-%dT%H:%M:%S")

    df = df.filter(
        (pl.col("datetime") >= start_datetime) & (pl.col("datetime") <= end_datetime)
    )

    # Aggregate to hourly data
    df = df.with_columns(
        pl.col("datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S")
        .dt.strftime("%Y-%m-%dT%H")
        .alias("datetime_hour")
    )

    df = df.group_by("datetime_hour").agg([
        pl.col("env_temp").mean().alias("avg_env_temp"),
        pl.col("sum_kwh_diff").sum().alias("total_kwh_diff")
    ]).sort(by="datetime_hour")

    print(f"Stats for {eco_u_data["1a9266de-dfff-11eb-9d61-02b402f0c1de"]["name"]}")
    print(f"All datapoints: {len(df)}")
    print(f"Start date: {start_datetime}")
    print(f"End date: {end_datetime}")
    print(f"Duration: {duration.days} days")
    print("\n")

    return duration.days, len(df)


if __name__ == '__main__':
    durations = list()
    number_of_datapoints = 0
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())

    for eco_u_id in list(eco_u_data.keys()):
        print(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['name']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_consumption.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_consumption.csv")
            dur, n = consumption_extractor(data=data_raw)
            durations.append(dur)
            number_of_datapoints += n
        else:
            print(f"Missing file for: {input_folder}/{eco_u_id}.csv")
    print(f"Number of sensors: {len(durations)}")
    print(f"Number of datapoints: {number_of_datapoints}")
    print(f"Average length of data sensor records: {sum(durations)/len(durations)}")