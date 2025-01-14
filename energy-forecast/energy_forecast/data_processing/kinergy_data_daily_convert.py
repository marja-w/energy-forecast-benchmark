import json
import os.path

import pandas as pd
import polars as pl

from src.energy_forecast.config import DATA_DIR

kinergy_folder = DATA_DIR / "kinergy"
input_folder = kinergy_folder / "consumption_data"
output_folder = DATA_DIR / "raw"
eco_u_data_file = kinergy_folder / "kinergy_eco_u_list.json"

orga_to_ort = {"BEM": "Berlin", "WWG": "Würzburg", "SBH": "Hamburg", "Jo-Sti": "Bamberg", "Ploen": "Ploen"}


def consumption_extractor(data: pd.DataFrame) -> pd.DataFrame:
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

    # merge info from json
    df["qmbehfl"] = eco_u_data[eco_u_id]["heated_area"]
    df["adresse"] = eco_u_data[eco_u_id]["name"]
    df["ort"] = orga_to_ort[eco_u_data[eco_u_id]["orga"]]

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

    print(f"Span from {df['date'].min()} to {df['date'].max()}")

    return df


if __name__ == '__main__':
    all_data = pd.DataFrame()
    with open(eco_u_data_file, "r", encoding="UTF-8") as f:
        eco_u_data = json.loads(f.read())

    for eco_u_id in list(eco_u_data.keys()):
        print(f"EcoU {eco_u_id} - {eco_u_data[eco_u_id]['name']}")

        if os.path.isfile(f"{input_folder}/{eco_u_id}_consumption.csv"):
            data_raw = pl.read_csv(f"{input_folder}/{eco_u_id}_consumption.csv")
            consumption_data = consumption_extractor(data=data_raw)
            print(f"Adding {len(consumption_data)} datapoints")
            print("\n")
            all_data = pd.concat([all_data, consumption_data])
        else:
            print(f"Missing file for: {input_folder}/{eco_u_id}.csv")

    all_data.to_csv(output_folder / "kinergy_daily_values.csv", index=False)
    print(f"Data saved to {output_folder}/kinergy_daily_values.csv")
    print(f"Total {len(all_data)} datapoints")
