import ast
import sys
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.io
import polars as pl
import wandb
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def plot_missing_dates(df_data: pl.DataFrame, res: str, sensor_id: str):
    df_data = df_data.filter(pl.col("id") == sensor_id)
    # address = df_data["adresse"].unique().item()
    freq = "D" if res == "daily" else "h"
    missing_dates: list[datetime.date] = get_missing_dates(df_data, freq).select(
        pl.col("missing_dates")).item().to_list()
    t_delta = timedelta(days=1) if res == "daily" else timedelta(hours=1)
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
    time_span = pd.date_range(min_date, max_date, freq=freq)

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
    output_dir = FIGURES_DIR / "missing_data" / res
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = output_dir / f"{sensor_id}.png"
    plt.savefig(output_file_path)
    logger.info(f"Saved missing dates plot to {output_file_path}")
    plt.close()


def plot_missing_dates_per_building(df: pl.DataFrame, res: str):
    logger.info(f"Plotting missing date plots to {FIGURES_DIR / 'missing_data'}")
    for (b_id, b_df) in df.group_by(["id"]):
        plot_missing_dates(b_df, res=res, sensor_id=b_id[0])


def plot_interpolated_series(series, b_id: str, data_source: str):
    # logger.info(f"Plotting interpolated series to {FIGURES_DIR / 'interpolated_data'}")
    fig, ax = plt.subplots()
    ax = series.plot()
    ax.set_title(data_source + " " + b_id)
    plt.savefig(FIGURES_DIR / "interpolated_data" / f"{b_id}.png")
    plt.close()


def plot_series(series, b_id: str, data_source: str, folder: Path):
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


def save_plot(plot_save_path: Path):
    # Save the plot as a PNG file
    plt.savefig(plot_save_path, format='png', bbox_inches='tight')
    logger.info(f"Saved plot to {plot_save_path}")
    plt.close()


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
        plot_dir = REPORTS_DIR / "predictions" / f"{model_name}_{n_out}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_save_path = plot_dir / f"{b_id}.png"
        save_plot(plot_save_path)
        logger.info(f"Plotted predictions for ID {b_id}")


def create_box_plot_predictions(id_to_metrics: list, metric_to_plot: str, run: Optional[wandb.sdk.wandb_run.Run],
                                log_y: bool = False, model_folder: str = ""):
    df = pd.DataFrame(id_to_metrics).explode(metric_to_plot)
    df[metric_to_plot] = df[metric_to_plot].astype(float)
    df = df.sort_values("avg_diff")

    fig = px.box(df, x="id", y=metric_to_plot, log_y=log_y, custom_data=["avg_diff"],
                 title=f"Boxplot of Prediction {metric_to_plot}")
    fig.update_traces(
        hovertemplate="<br>".join([
            "id: %{x}",
            metric_to_plot + ": %{y}",
            "avg_diff: %{customdata[0]}"
        ])
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=f"{metric_to_plot} (kwh) {'(log)' if log_y else ''}"
            )
        )
    )
    if run:
        if sys.platform == "win32":
            wandb.log({"boxplot_predictions": wandb.Html(plotly.io.to_html(fig))})
        else:
            run.log({f"boxplot_predictions": fig})
    else:
        fig.show()
        # fig.write_html(f"{model_folder}/boxplot_{metric_to_plot}.html")


def create_box_plot_predictions_by_size(id_to_metrics: list, metric_to_plot: str, entry_threshold: int,
                                        run: Optional[wandb.sdk.wandb_run.Run],
                                        log_y: bool = False):
    df = pd.DataFrame(id_to_metrics).explode(metric_to_plot)

    # Count entries per ID and add size column
    entry_counts = df.groupby('id').size()
    df['size'] = df['id'].map(lambda x: 1 if entry_counts[x] >= entry_threshold else 0)
    class_sizes = df["size"].value_counts()
    df[metric_to_plot] = df[metric_to_plot].astype(float)

    # Create separate plots for each size
    fig = px.box(df, x="size", y=metric_to_plot, color="size", log_y=log_y, custom_data=["avg_diff", "id"],
                 title=f"Boxplot of Prediction {metric_to_plot} by Data Size ({entry_threshold}+ entries)",
                 color_discrete_map={0: "blue", 1: "red"})

    new_names = {"0": f"small ({class_sizes[0]})", "1": f"large ({class_sizes[1]})"}
    fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))

    fig.update_traces(
        hovertemplate="<br>".join([
            "id: %{customdata[1]}",
            metric_to_plot + ": %{y}",
            "avg_diff: %{customdata[0]}"
        ])
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=f"{metric_to_plot} (kwh) {'(log)' if log_y else ''}"
            )
        )
    )

    if run:
        if sys.platform == "win32":
            wandb.log({
                "boxplot_predictions_by_size": wandb.Html(plotly.io.to_html(fig))
            })
        else:
            run.log({
                "boxplot_predictions_by_size": fig
            })
    else:
        fig.show()


def plot_box_plot_hours(rse_list: list, hour_col: list, b_id: str, run: Optional[wandb.sdk.wandb_run.Run],
                                log_y: bool = False):
    df = pd.DataFrame({"rse": rse_list, "hour": hour_col})

    fig = px.box(df, x="hour", y="rse", log_y=log_y, title=f"Boxplot of RSE for {b_id} per Hour")
    fig.update_traces(
        hovertemplate="<br>".join([
            "hour: %{x}",
            "rse: %{y}"
        ])
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=f"RSE (kwh) {'(log)' if log_y else ''}"
            )
        )
    )
    if run:
        if sys.platform == "win32":
            wandb.log({"boxplot_hours": wandb.Html(plotly.io.to_html(fig))})
        else:
            run.log({f"boxplot_hours": fig})
    else:
        fig.show()


def plot_bar_chart(id_to_metrics: list, metric_to_plot: str, run: Optional[wandb.sdk.wandb_run.Run],
                   log_y: bool = False, name: str = ""):
    logger.info(f"Plotting Bar Chart for {name}")
    df = pd.DataFrame(id_to_metrics)
    df.rename(columns={"id": name}, inplace=True)

    fig = px.bar(df, x=name, y=metric_to_plot, log_y=log_y, color="avg_diff", custom_data=["avg_diff", "n_entries"],
                 title=f"Bar Chart for {metric_to_plot} ({name})")
    fig.update_traces(
        hovertemplate="<br>".join([
            name + ": %{x}",
            metric_to_plot + ": %{y}",
            "avg_diff: %{customdata[0]}",
            "number of entries: %{customdata[1]}"
        ])
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=f"{metric_to_plot} (kwh) {'(log)' if log_y else ''}"
            )
        )
    )
    if run:
        if sys.platform == "win32":
            wandb.log({f"bar_chart_{name}_{metric_to_plot}": wandb.Html(plotly.io.to_html(fig))})
        else:
            run.log({f"bar_chart_{name}_{metric_to_plot}": fig})
    else:
        fig.show()
        # fig.write_html(f"{model_folder}/boxplot_{metric_to_plot}.html")

def plot_box_plot(id_to_ind_metrics: list, metric_to_plot: str, run: Optional[wandb.sdk.wandb_run.Run],
                  log_y: bool = False, name: str = ""):
    logger.info(f"Plotting Box Plot for {name}")
    df = pd.DataFrame(id_to_ind_metrics).explode(metric_to_plot)
    df[metric_to_plot] = df[metric_to_plot].astype(float)
    df = df.sort_values("avg_diff")

    df.rename(columns={"id": name}, inplace=True)

    fig = px.box(df, x=name, y=metric_to_plot, log_y=log_y, custom_data=["avg_diff"],
                 title=f"Boxplot of Prediction {metric_to_plot}")
    fig.update_traces(
        hovertemplate="<br>".join([
            "id: %{x}",
            metric_to_plot + ": %{y}",
            "avg_diff: %{customdata[0]}"
        ])
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=f"{metric_to_plot} (kwh) {'(log)' if log_y else ''}"
            )
        )
    )
    if run:
        if sys.platform == "win32":
            wandb.log({f"box_plot_{name}_{metric_to_plot}": wandb.Html(plotly.io.to_html(fig))})
        else:
            run.log({f"box_plot_{name}_{metric_to_plot}": fig})
    else:
        fig.show()

def plot_multiple_model_metrics(metrics: list[dict]):
    """
        Create a bar chart for model metrics where each value in the list gets its own bar.

        Args:
            df: A Polars DataFrame with columns 'model', 'metric', and 'val' where 'val' contains lists.

        Returns:
            A Plotly figure object with the bar chart.
        """
    # Validate input DataFrame
    df = pd.DataFrame(metrics)
    required_cols = ["model", "metric", "val"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Check if 'val' contains lists
    if not isinstance(df["val"].iloc[0], (list, np.ndarray)):
        raise ValueError("The 'val' column must contain lists or arrays")

    # Get the length of the lists
    list_lengths = df["val"].apply(len)
    if len(list_lengths) == 0:
        raise ValueError("DataFrame is empty or contains no valid lists")

    # Ensure all lists have the same length
    if not (list_lengths == list_lengths.iloc[0]).all():
        raise ValueError("All lists in 'val' must have the same length")

    # Create an exploded DataFrame
    exploded_df = pd.DataFrame()

    for idx, row in df.iterrows():
        for step, val in enumerate(row["val"]):
            new_row = {
                "model": row["model"],
                "metric": row["metric"],
                "val": val,
                "step": step
            }
            exploded_df = pd.concat([exploded_df, pd.DataFrame([new_row])], ignore_index=True)

    # Get unique models and metrics
    models = exploded_df["model"].unique()
    metrics = exploded_df["metric"].unique()
    steps = exploded_df["step"].unique()

    # Create figure with subplots (one per metric)
    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        subplot_titles=[f"Metric: {metric}" for metric in metrics],
        vertical_spacing=0.1
    )

    df = exploded_df
    # Add traces for each model and metric
    for i, metric in enumerate(metrics):
        metric_df = df[df["metric"] == metric]

        for model in models:
            model_data = metric_df[metric_df["model"] == model]

            fig.add_trace(
                go.Bar(
                    x=model_data["step"],
                    y=model_data["val"],
                    name=model,
                    legendgroup=model,
                    showlegend=(i == 0),  # Only show in legend once
                ),
                row=i + 1,
                col=1
            )

    # Update layout
    fig.update_layout(
        title="Model Metrics Comparison",
        barmode="group",
        height=300 * len(metrics),
        legend_title="Models",
        xaxis_title="Step",
    )

    # Update y-axes titles
    for i in range(len(metrics)):
        fig.update_yaxes(title_text="Value", row=i + 1, col=1)
        fig.update_xaxes(title_text="Step", row=i + 1, col=1)

    return fig


def plot_filtered_data_points(df, column, filtered_df):
    """Plot scatter plot comparing data points between two DataFrames.

    Args:
        df: Main DataFrame
        column: Column name to plot
        filtered_df: Filtered DataFrame to compare against
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get points present in both DataFrames
    common_points = df.filter(pl.col("datetime").is_in(filtered_df["datetime"]))

    # Get points only in df
    unique_points = df.filter(~pl.col("datetime").is_in(filtered_df["datetime"]))

    # Plot points
    ax.scatter(common_points["datetime"], common_points[column],
               color='blue', label='Data Points')
    ax.scatter(unique_points["datetime"], unique_points[column],
               color='red', label='Outlier')

    plt.xlabel('DateTime')
    plt.ylabel(column)
    sensor_id = df.get_column("id").unique()[0]
    plt.title(f'Outlier IQR Method - {sensor_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_dir = REPORTS_DIR / "figures" / f"outlier" / "daily"
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = plot_dir / f"{sensor_id}.png"
    save_plot(plot_save_path)
    logger.info(f"Plotted outlier for ID {sensor_id}")


def plot_reduced_data_eval():
    df = pd.read_csv(REPORTS_DIR / "reduced_data_eval_1_day_forecast.csv")

    # Create the line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='remove_per', y='test_rmse', hue='model', marker='o')

    # Add labels and title
    plt.xlabel('Percentage of Data Removed (%)')
    plt.ylabel('Test RMSE')
    plt.title('Model Performance vs. Data Removal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models')

    # Show the plot
    plt.tight_layout()
    # plt.show()

    plt.savefig(REPORTS_DIR / "figures" / "reduced_data_eval_1_day_forecast.png")


def plot_metrics_per_step():
    save_path = FIGURES_DIR / "metrics_per_step_box_plot.png"
    df = pd.read_csv(REPORTS_DIR / "best_models_7_day_forecast.csv")

    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        name = row['name']
        mae_values = ast.literal_eval(row['summary']).get('test_mae_ind', [])
        avg_mae = np.mean(mae_values)

        if not mae_values:
            print(f"Warning: No 'test_mae_ind' data found for {name}")
            continue
        steps = np.arange(len(mae_values))
        line, = plt.plot(steps, mae_values, label=f"{name} (avg: {avg_mae:.4f})",
                         marker='o', markersize=3)

        # Add annotation at the end of the line with the average value
        last_x = steps[-1]
        last_y = mae_values[-1]

        # Slight offset for better visibility
        plt.annotate(
            f"avg: {avg_mae:.4f}",
            xy=(last_x, last_y),
            xytext=(last_x + 0.2, last_y),
            fontsize=9,
            color=line.get_color(),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=line.get_color()),
            arrowprops=dict(arrowstyle="->", color=line.get_color())
        )

    plt.xlabel('Step')
    plt.ylabel('MAE')
    plt.title("Per-Step Metric Evaluation of Best Models")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_box_plot_per_step():
    # save_path = None
    save_path = FIGURES_DIR / "metrics_per_step.png"
    df = pd.read_csv(REPORTS_DIR / "best_models_7_day_forecast.csv")
    # Prepare data for boxplot
    box_data = []
    labels = []
    avg_values = []

    for _, row in df.iterrows():
        name = row['name']
        mae_values = ast.literal_eval(row['summary']).get('test_mae_ind', [])

        if not mae_values:
            print(f"Warning: No 'test_mae_ind' data found for {name}")
            continue

        box_data.append(mae_values)
        labels.append(name)
        avg_values.append(np.mean(mae_values))

    # Create the boxplot
    box_plot = plt.boxplot(box_data, patch_artist=True, labels=labels)

    # Customize boxplot appearance
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(box_data)))

    for i, (box, color) in enumerate(zip(box_plot['boxes'], colors)):
        box.set(facecolor=color, alpha=0.7)

        # Add average value annotation above each box
        avg = avg_values[i]
        plt.annotate(
            f"avg: {avg:.4f}",
            xy=(i + 1, box_plot['caps'][i * 2].get_ydata()[0]),  # Position above the upper cap
            xytext=(0, 10),  # Offset text 10 points above
            textcoords='offset points',
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec='gray'),
        )

    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.title("Box Plot of Per-Step Metric Evaluation of Best Models")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add some padding to y-axis to accommodate annotations
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    # df_daily = pl.read_csv(PROCESSED_DATA_DIR / "dataset_daily.csv").with_columns(pl.col("datetime").str.to_datetime())
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
    # for id in df_daily["id"].unique():
        # plot_missing_dates(df_daily, id)
    # plot_reduced_data_eval()
    plot_box_plot_per_step()