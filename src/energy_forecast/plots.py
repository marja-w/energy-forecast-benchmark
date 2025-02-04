from datetime import timedelta, datetime, date

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt


def get_missing_dates(df: pl.DataFrame, primary_energy: str, frequency: str):
    missing_dates = []
    df = df.with_columns(pl.col("date").dt.date().alias("date"))
    df_time = df.filter(pl.col('primary_energy') == primary_energy).group_by(pl.col("id")).agg(pl.len(),
                                                                                               pl.col(
                                                                                                   "date").min().alias(
                                                                                                   "min_date"),
                                                                                               pl.col(
                                                                                                   "date").max().alias(
                                                                                                   "max_date"))
    for row in df_time.iter_rows():
        id = row[0]
        # print("\nSensor: ", id, "\n")
        # print(f"Sensor {id} has {row[1]} datapoints")
        start_date = row[2]
        end_date = row[3]

        date_list_rec = df.filter(pl.col("id") == id).select(pl.col("date"))["date"].to_list()

        date_list = pd.date_range(start_date, end_date, freq=frequency).date

        missing_dates_sensor = list(set(date_list) - set(date_list_rec))
        missing_dates_sensor.sort()
        logger.info(f"Missing dates for sensor {id}: {len(missing_dates_sensor)}")
        missing_dates.append(
            {"id": id, "missing_dates": missing_dates_sensor, "len": len(missing_dates_sensor), "n": row[1],
             "per": (((row[1] + len(missing_dates_sensor)) / (row[1])) * 100) - 100})
    return pl.DataFrame(missing_dates).sort(pl.col("len"), descending=True)


def find_time_spans(dates: list[date]) -> list[tuple[date, date, int]]:
    spans = []
    start_date = dates[0]
    prev_date = start_date
    count = 1

    for i in range(1, len(dates)):
        current_date = dates[i]

        if current_date == prev_date + timedelta(days=1):
            count += 1
        else:
            spans.append((start_date, prev_date, count))
            start_date = current_date
            count = 1

        prev_date = current_date

    spans.append((start_date, prev_date, count))  # Add the last span
    return spans


def plot_missing_dates(df: pl.DataFrame, sensor_id: str):
    missing_dates = get_missing_dates(df, "district heating", "D").filter(pl.col("id") == sensor_id).select(
        pl.col("missing_dates")).item().to_list()
    spans = find_time_spans(missing_dates)
    min_date = df.filter(pl.col('id') == sensor_id).select(pl.col("date").min()).item()
    max_date = df.filter(pl.col('id') == sensor_id).select(pl.col("date").max()).item()
    time_span = pd.date_range(min_date, max_date, freq="D").date

    df = pl.DataFrame({"date": list(time_span)})
    df = df.join(df.filter(pl.col("id") == sensor_id), on="date", how="left")

    df = df.to_pandas()
    df = df.set_index('date')

    fig, ax = plt.subplots()
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=df["diff"], facecolor="lightblue", alpha=0.5)
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=np.isfinite(df["diff"]), facecolor="white",
                    alpha=1)
    ax.plot(df.index, df["diff"])

    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
