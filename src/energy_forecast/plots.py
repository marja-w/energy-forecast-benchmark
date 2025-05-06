from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional

import darts
import numpy as np
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
import os

from src.energy_forecast.config import PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR
from src.energy_forecast.utils.util import find_time_spans, get_missing_dates, store_plot_wandb


def plot_means(X_train, y_train, X_val, y_val, X_test, y_test):
    fig, ax = plt.subplots()
    bar_width = 0.25
    ax.set_title(f"Mean values of data")
    columns = list(X_train.columns) + y_train.columns.tolist()

    br1 = np.arange(len(columns))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]

    d = pd.concat([X_train, y_train])
    _mean = d.mean()
    _min = d.min()
    _max = d.max()
    # yerr = [np.subtract(_mean, _min), np.subtract(_max, _mean)]
    plt.bar(br1, _mean, label=f"Training data ({len(X_train)})", width=bar_width, capsize=10)
    plt.bar(br2, pd.concat([X_val.mean(), y_val.mean()]), label=f"Validation data ({len(X_val)})", width=bar_width)
    plt.bar(br3, pd.concat([X_test.mean(), y_test.mean()]), label=f"Test data ({len(X_test)})", width=bar_width)
    plt.xticks([r + bar_width for r in range(len(columns))], columns)
    plt.legend()
    # plt.ion()  # interactive mode non blocking
    plt.show()


def plot_std(X_train, y_train, X_val, y_val, X_test, y_test):
    fig, ax = plt.subplots()
    bar_width = 0.25
    ax.set_title(f"Std values of data")
    columns = list(X_train.columns) + y_train.columns.tolist()

    br1 = np.arange(len(columns))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]

    plt.bar(br1, pd.concat([X_train.std(), y_train.std()]), label=f"Training data ({len(X_train)})", width=bar_width)
    plt.bar(br2, pd.concat([X_val.std(), y_val.std()]), label=f"Validation data ({len(X_val)})", width=bar_width)
    plt.bar(br3, pd.concat([X_test.std(), y_test.std()]), label=f"Test data ({len(X_test)})", width=bar_width)
    plt.xticks([r + bar_width for r in range(len(columns))], columns)
    plt.legend()
    # plt.ion()
    plt.show()


def plot_train_val_test_split(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    for b_id in train_df["id"].unique():
        t = train_df.filter(pl.col("id") == b_id).select(["datetime", "diff"]).with_columns(
            pl.lit("train").alias("split"))
        v = val_df.filter(pl.col("id") == b_id).select(["datetime", "diff"]).with_columns(pl.lit("val").alias("split"))
        te = test_df.filter(pl.col("id") == b_id).select(["datetime", "diff"]).with_columns(
            pl.lit("test").alias("split"))
        df = pl.concat([t, v, te])
        fig, ax = plt.subplots(figsize=(12, 4), dpi=80)
        # ax.plot(list(df["datetime"]), list(df["diff"]), c=list(df["split"]))
        # plt.show()
        sns.scatterplot(data=df, x="datetime", y="diff", hue="split")
        ax.set_title(b_id)
        plt.show()
    raise ValueError
    # df.plot.line(x="datetime", y="diff", color="split")


def plot_missing_dates(df_data: pl.DataFrame, sensor_id: str):
    df_data = df_data.filter(pl.col("id") == sensor_id)
    # address = df_data["adresse"].unique().item()
    missing_dates: list[datetime.date] = get_missing_dates(df_data, "D").select(
        pl.col("missing_dates")).item().to_list()
    df_spans: pl.DataFrame = find_time_spans(missing_dates, delta=timedelta(days=1))
    if df_spans.is_empty():
        avg_length = 0
        n_spans = 0
    else:
        avg_length = df_spans["n"].mean()
        n_spans = len(df_spans)

    logger.info(f"{sensor_id} average length of missing dates: {avg_length}")
    logger.info(f"{sensor_id} number of missing time spans: {n_spans}")
    min_date = df_data.select(pl.col("datetime").min()).item()
    max_date = df_data.select(pl.col("datetime").max()).item()
    time_span = pd.date_range(min_date, max_date, freq="D")

    df = pl.DataFrame({"datetime": list(time_span)})
    df = df.join(df_data, on="datetime", how="left")

    df = df.to_pandas()
    df = df.set_index('datetime')

    # plot missing dates
    fig, ax = plt.subplots()
    source_code = df_data["source"].unique().item()
    ax.set_title(source_code + " " + sensor_id)
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=df["diff"], facecolor="lightblue", alpha=0.5)
    ax.fill_between(df.index, df["diff"].min(), df["diff"].max(), where=np.isfinite(df["diff"]), facecolor="white",
                    alpha=1)
    ax.scatter(df.index, df["diff"])

    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_data" / f"{sensor_id}.png")
    plt.close()


def plot_missing_dates_per_building(df: pl.DataFrame):
    logger.info(f"Plotting missing date plots to {FIGURES_DIR / 'missing_data'}")
    for (b_id, b_df) in df.group_by(["id"]):
        plot_missing_dates(b_df, sensor_id=b_id[0])


def plot_interpolated_series(series: darts.TimeSeries, b_id: str, data_source: str):
    # logger.info(f"Plotting interpolated series to {FIGURES_DIR / 'interpolated_data'}")
    fig, ax = plt.subplots()
    ax = series.plot()
    ax.set_title(data_source + " " + b_id)
    plt.savefig(FIGURES_DIR / "interpolated_data" / f"{b_id}.png")
    plt.close()


def plot_series(series: darts.TimeSeries, b_id: str, data_source: str, folder: Path):
    # logger.info(f"Plotting interpolated series to {FIGURES_DIR / 'interpolated_data'}")
    plt.subplots()
    ax = series.plot()
    ax.set_title(data_source + " " + b_id)
    plt.savefig(folder / f"{b_id}.png")
    plt.close()


def plot_dataframe(df: pl.DataFrame, b_id: str, data_source: str, folder: Path):
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="datetime", y="diff")
    ax.set_title(data_source + " " + b_id)
    plt.savefig(folder / f"{b_id}.png")
    plt.close()


def plot_clusters(df: pl.DataFrame, labels: np.ndarray):
    fig, ax = plt.subplots()
    sns.scatterplot(
        pl.concat([df, pl.DataFrame(labels, schema=["label"])], how="horizontal").to_pandas(),
        x="avg",
        y="std",
        hue="label",
        ax=ax,
    )
    plt.savefig(FIGURES_DIR / "clusters.png")
    logger.success("Cluster plots saved")


def plot_per_step_metrics(per_step_metrics: np.ndarray):
    num_rows = per_step_metrics.shape[0]
    bar_width = 0.35
    indices = np.arange(num_rows)

    # Plot the bars for each column in the data
    plt.bar(indices, per_step_metrics[:, 0], bar_width, label='Absolute Error')
    plt.bar(indices + bar_width, per_step_metrics[:, 1], bar_width, label='Squared Log Error')

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.title('Per Time Step Metrics')
    plt.xticks(indices + bar_width / 2, [f'Step {i}' for i in range(num_rows)])
    plt.legend()
    plt.savefig(FIGURES_DIR / "per_step_metrics.png")
    store_plot_wandb(plt, "per_step_metrics.png")


def plot_predictions(ds, y: np.ndarray, b_id: str, y_hat: np.ndarray, dates: pl.Series, lag_in: int, n_out: int,
                     lag_out: int, run: Optional[wandb.sdk.wandb_run.Run], model_name: str):
    test_df = ds.get_test_df().filter(pl.col("id") == b_id)
    plt.figure(figsize=(10, 6))

    if y_hat.shape[1] == 1:
        plt.plot(dates, y, label='Test')
        y_hat = y_hat.squeeze(1)
        plt.scatter(dates, y_hat, linewidth=2, color='red')
    else:
        plt.plot(test_df["datetime"], test_df["diff"], label='Test')
        test_dates = test_df[lag_in:]["datetime"]
        for row_id, row in enumerate(y_hat):
            plt.plot(test_dates[row_id:row_id + n_out], row, linewidth=2, color='red')

    plt.xlabel('Time (Day)')
    plt.ylabel('Value')
    plt.title('Time Series Forecast vs Actual Series')
    plt.legend()
    plt.grid(True)

    if run:
        return plt  # store plot to wandb.Table
    else:
        # Save the plot as a PNG file
        plot_dir = REPORTS_DIR / "predictions" / f"{model_name}_{lag_in}_{n_out}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_save_path = plot_dir / f"{b_id}.png"
        plt.savefig(plot_save_path, format='png', bbox_inches='tight')
        logger.info(f"Plotted predictions for ID {b_id}")
        plt.close()


if __name__ == "__main__":
    df_daily = pl.read_csv(PROCESSED_DATA_DIR / "dataset_daily.csv").with_columns(pl.col("datetime").str.to_datetime())
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
    # df_daily_dh = df_daily.filter(pl.col("source") == "dh")
    for id in df_daily["id"].unique():
        plot_missing_dates(df_daily, id)
