from datetime import timedelta, datetime, date

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt

from src.energy_forecast.config import RAW_DATA_DIR
from src.energy_forecast.util import find_time_spans, get_missing_dates


def plot_missing_dates(df_data: pl.DataFrame, sensor_id: str):
    df_data = df_data.filter(pl.col("id") == sensor_id)
    address = df_data["adresse"].unique().item()
    missing_dates: list[datetime.date] = get_missing_dates(df_data, "D").select(
        pl.col("missing_dates")).item().to_list()
    df_spans: pl.DataFrame = find_time_spans(missing_dates)
    if df_spans.is_empty():
        avg_length = 0
        n_spans = 0
    else:
        avg_length = df_spans["n"].mean()
        n_spans = len(df_spans)

    logger.info(f"{sensor_id} average length of missing dates: {avg_length}")
    logger.info(f"{sensor_id} number of missing time spans: {n_spans}")
    min_date = df_data.select(pl.col("date").min()).item()
    max_date = df_data.select(pl.col("date").max()).item()
    time_span = pd.date_range(min_date, max_date, freq="D").date

    df = pl.DataFrame({"date": list(time_span)})
    df = df.join(df_data, on="date", how="left")

    df = df.to_pandas()
    df = df.set_index('date')

    # plot missing dates
    fig, ax = plt.subplots()
    ax.set_title(address + " " + sensor_id)
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=df["diff"], facecolor="lightblue", alpha=0.5)
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=np.isfinite(df["diff"]), facecolor="white",
                    alpha=1)
    ax.plot(df.index, df["diff"])

    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_daily = pl.read_csv(RAW_DATA_DIR / "daily.csv").with_columns(pl.col("date").str.to_date())
    # ids = df["id"].unique()
    corrupt_sensors = ["""d566a120-d232-489a-aa42-850e5a44dbee""",
                       """7dd30c54-3be7-4a3c-b5e0-9841bb3ffddb""",
                       """5c8f03f4-9165-43a2-8c42-1e813326934e""",
                       """4ccc1cea-534d-4dbe-bf66-0d31d887088e""",
                       """5e2fd59d-603a-488b-a525-513541039c4a""",
                       """8ff79953-ad51-40b5-a025-f24418cae4b1""",
                       """4f36b3bd-337e-4b93-9333-c53a28d0c417""",
                       """2b9a3bc7-252f-4a10-8ccb-5ccce53e896a""",
                       """44201958-2d6b-4952-956c-22ea951a6442""",
                       """1a94c658-a524-4293-bb95-020c53beaabd""",
                       """0c9ad311-b86f-4371-a695-512ca49c70a7""",
                       """2f025f96-af2c-4140-b955-766a791fa925""",
                       """8f7b3862-a50d-44eb-8ac9-de0cf48a6bd2""",
                       """d5fb4343-04d4-4521-8a4b-feaf772ff376""",
                       """35d897c4-9486-41c1-be9b-0e1707d9fbef""",
                       """a9644794-439b-401c-b879-8c0225e16b99""",
                       """61470655-33c1-4717-b729-baa6658a6aeb""",
                       """bc098a2e-0cc7-4f01-b6ad-9d647ae9f627""",
                       """b6b63b91-da14-449d-b213-e6ef5ca27e67""",
                       """573a7d1e-de3f-49e1-828b-07d463d1fa4d"""
                       ]
    df_daily_dh = df_daily.filter(pl.col("source") == "dh")
    for id in df_daily_dh["id"].unique():
        plot_missing_dates(df_daily, id)
