import os
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from wandb.sdk.wandb_run import Run

from src.energy_forecast.config import DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.energy_forecast.dataset import TrainingDataset
from src.energy_forecast.plots import plot_bar_chart, plot_predictions, create_box_plot_predictions, \
    plot_box_plot_hours, plot_box_plot
from src.energy_forecast.utils.metrics import get_metrics, get_mean_metrics


def calculate_metrics_per_id(ds: TrainingDataset, run: Optional[Run], config: dict, model_name: str, plot: bool = False) -> None:
    """
    Calculates regression metrics for each unique identifier in the dataset.

    This method evaluates the prediction performance of the model on a test dataset
    provided in `ds`. It computes various regression metrics such as MAPE, MSE, MAE,
    NRMSE, RMSE, and average difference. The results are logged or visualized based
    on the input parameters, and optionally saved for further analysis.

    :param ds: An instance of `TrainingDataset` with configurations, test datasets,
        scalers, and related data required for metrics calculation.
    :type ds: TrainingDataset

    :param run: An optional instance of `Run`, used to log visualizations and metrics
        to `wandb` dashboard. If `None`, the metrics are saved locally instead.
    :type run: Optional[Run]

    :param plot: A boolean flag indicating whether to generate and include prediction
        plots in the computed outputs. Set to `True` to enable plotting.
    :type plot: bool
    """
    id_to_metrics = list()
    id_to_ind_metrics = list()
    if run:
        table_cols = ["id", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        if plot:
            table_cols = ["id", "predictions", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)

    b_id_array = ds.id_column
    y_test = ds.y_test.to_numpy()
    pred_list = list()
    for b_id in ds.id_column.unique():
        id_mask = b_id_array == b_id
        date_mask = ds.id_column == b_id
        y_hat = ds.y_hat[id_mask]
        y = y_test[id_mask]
        heated_area = ds.get_heated_area_by_id(b_id)
        pred_list.append({"id": b_id, "predictions": y_hat})

        test_mse, test_mae, test_rmse, test_nrmse, test_mape, rse_list = get_metrics(y, y_hat)
        mean_target_value = np.mean(y)

        id_to_ind_metrics.append(
            {"id": b_id, "rse": rse_list, "nrse": rse_list / heated_area, "avg_diff": mean_target_value})

        # mean over metrics if output length > 1
        if y.shape[1] > 1:
            test_mse, test_mae, test_rmse, test_nrmse, test_mape = get_mean_metrics(test_mse, test_mae, test_rmse,
                                                                                    test_nrmse, test_mape)

        if plot:
            plt = plot_predictions(ds, y, b_id, y_hat, ds.datetime_column.filter(date_mask), config["lag_in"], config["n_out"],
                                   config["lag_out"], run, model_name)
        if run and plot:
            if plt:
                wandb_table.add_data(b_id, wandb.Image(plt), test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                     mean_target_value)
                plt.close()
        # elif run:  # problems with WandB table not showing
        #     wandb_table.add_data(b_id, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
        #                          mean_target_value)
        elif not run:
            b_metrics = {"id": b_id, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                         "rmse": test_rmse, "avg_diff": mean_target_value}
            id_to_metrics.append(b_metrics)
    # store predictions for each building
    df_predictions = pd.DataFrame(pred_list)
    output_dir = REPORTS_DIR / "predictions" / f"{model_name}_{config['n_out']}_{wandb.run.id}"
    os.makedirs(output_dir, exist_ok=True)
    csv_ = output_dir / "predictions.csv"
    df_predictions.to_csv(csv_, index=False)
    logger.info(f"Predictions saved to {csv_}")

    create_box_plot_predictions(id_to_ind_metrics, "rse", run, log_y=True)
    create_box_plot_predictions(id_to_ind_metrics, "nrse", run, log_y=True)
    # create_box_plot_predictions_by_size(id_to_ind_metrics, "rse", 50, run, log_y=False)
    # create_box_plot_predictions_by_size(id_to_ind_metrics, "rse", 100, run, log_y=False)

    metrics_df = pl.DataFrame(id_to_metrics)
    if run:
        run.log({"building_metrics": wandb_table})
    else:
        metrics_df.write_csv(
            REPORTS_DIR / "metrics" / f"{model_name}_{config['res']}_{config['n_out']}.csv")  # overwrites in next run
        logger.info(f"Metrics saved to {REPORTS_DIR}/metrics/{model_name}_{config['n_out']}.csv")


def calculate_metrics_per_id_and_hour(ds: TrainingDataset, run: Optional[Run], config: dict, model_name: str,
                             plot: bool = False) -> None:
    id_to_metrics = list()
    id_to_ind_metrics = list()
    if run:
        table_cols = ["id", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        if plot:
            table_cols = ["id", "predictions", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)

    hours_array = ds.datetime_column.dt.hour().to_numpy()
    b_id_array = ds.id_column
    y_test = ds.y_test.to_numpy()
    for b_id in ds.id_column.unique():
        id_mask = b_id_array == b_id
        date_mask = ds.id_column == b_id
        y_hat = ds.y_hat[id_mask]
        y = y_test[id_mask]
        hours_id = hours_array[id_mask]
        heated_area = ds.get_heated_area_by_id(b_id)

        test_mse, test_mae, test_rmse, test_nrmse, test_mape, rse_list = get_metrics(y, y_hat)
        mean_target_value = np.mean(y)

        plot_box_plot_hours(rse_list, hours_id, b_id, run, log_y=False)

        id_to_ind_metrics.append(
            {"id": b_id, "rse": rse_list, "nrse": rse_list / heated_area, "avg_diff": mean_target_value})

        # mean over metrics if output length > 1
        if y.shape[1] > 1:
            test_mse, test_mae, test_rmse, test_nrmse, test_mape = get_mean_metrics(test_mse, test_mae, test_rmse,
                                                                                    test_nrmse, test_mape)

        if plot:
            plt = plot_predictions(ds, y, b_id, y_hat, ds.datetime_column.filter(date_mask), config["lag_in"],
                                   config["n_out"],
                                   config["lag_out"], run, model_name)
        if run and plot:
            wandb_table.add_data(b_id, wandb.Image(plt), test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
            plt.close()
        elif run:
            wandb_table.add_data(b_id, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
        elif not run:
            b_metrics = {"id": b_id, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                         "rmse": test_rmse, "avg_diff": mean_target_value}
            id_to_metrics.append(b_metrics)
    create_box_plot_predictions(id_to_ind_metrics, "rse", run, log_y=True)
    create_box_plot_predictions(id_to_ind_metrics, "nrse", run, log_y=True)
    # create_box_plot_predictions_by_size(id_to_ind_metrics, "rse", 50, run, log_y=False)
    # create_box_plot_predictions_by_size(id_to_ind_metrics, "rse", 100, run, log_y=False)

    metrics_df = pl.DataFrame(id_to_metrics)
    if run:
        run.log({"building_metrics": wandb_table})
    else:
        metrics_df.write_csv(
            REPORTS_DIR / "metrics" / f"{model_name}_{config['n_out']}.csv")  # overwrites in next run
        logger.info(f"Metrics saved to {REPORTS_DIR}/metrics/{model_name}_{config['n_out']}.csv")

def calculate_metrics_per_month(ds: TrainingDataset, run=None, plot: bool = False) -> None:
    id_to_metrics = list()
    id_to_ind_metrics = list()
    meta_df = pl.read_csv(
        PROCESSED_DATA_DIR / f"dataset_interpolate_{ds.config['res']}_feat.csv")  # for retrieving heated area
    if run:
        table_cols = ["month", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)

    month_array = ds.datetime_column.dt.month().to_numpy()
    y_test = ds.y_test.to_numpy()
    for month in range(1, 13):
        month_mask = month_array == month
        y_hat = ds.y_hat[month_mask]
        y = y_test[month_mask]

        test_mse, test_mae, test_rmse, test_nrmse, test_mape, rse_list = get_metrics(y, y_hat)

        mean_target_value = np.mean(y)
        id_to_ind_metrics.append(
            {"id": month, "rse": rse_list, "nrse": rse_list / mean_target_value, "avg_diff": mean_target_value})

        # mean over metrics if output length > 1
        if y.shape[1] > 1:
            test_mse, test_mae, test_rmse, test_nrmse, test_mape = get_mean_metrics(test_mse, test_mae, test_rmse,
                                                                                    test_nrmse, test_mape)

        if run:
            wandb_table.add_data(month, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
        curr_metrics = {"id": month, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                        "rmse": test_rmse, "avg_diff": mean_target_value, "n_entries": len(y)}
        id_to_metrics.append(curr_metrics)

    plot_bar_chart(id_to_metrics, "rmse", run, log_y=False, name="month")

def calculate_metrics_per_hour(ds: TrainingDataset, run=None, plot: bool = False) -> None:
    meta_df = pl.read_csv(
        PROCESSED_DATA_DIR / f"dataset_interpolate_{ds.config['res']}_feat.csv")  # for retrieving heated area
    if run:
        table_cols = ["month", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)

    hours_array = ds.datetime_column.dt.hour().to_numpy()
    y_test = ds.y_test.to_numpy()
    id_to_metrics = list()
    id_to_ind_metrics = list()
    for hour in range(1, 25):
        hour_mask = hours_array == hour
        y_hat = ds.y_hat[hour_mask]
        y = y_test[hour_mask]

        test_mse, test_mae, test_rmse, test_nrmse, test_mape, rse_list = get_metrics(y, y_hat)

        mean_target_value = np.mean(y)
        id_to_ind_metrics.append(
            {"id": hour, "rse": rse_list, "nrse": rse_list / mean_target_value, "avg_diff": mean_target_value})

        # mean over metrics if output length > 1
        if y.shape[1] > 1:
            test_mse, test_mae, test_rmse, test_nrmse, test_mape = get_mean_metrics(test_mse, test_mae, test_rmse,
                                                                                    test_nrmse, test_mape)

        if run:
            wandb_table.add_data(hour, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
        curr_metrics = {"id": hour, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                        "rmse": test_rmse, "avg_diff": mean_target_value, "n_entries": len(y)}
        id_to_metrics.append(curr_metrics)

    plot_bar_chart(id_to_metrics, "rmse", run, log_y=False, name="hour")
    plot_bar_chart(id_to_metrics, "nrmse", run, log_y=False, name="hour")
    plot_box_plot(id_to_ind_metrics, "rse", run, log_y=False, name="hour")
    plot_box_plot(id_to_ind_metrics, "nrse", run, log_y=False, name="hour")


def normalize_by_avg_per_id(y_hat, id_column):
    n_samples, n_features = y_hat.shape

    if len(id_column) != n_samples:
        raise ValueError(
            f"id_column length ({len(id_column)}) must match "
            f"y_hat first dimension ({n_samples})"
        )

    # Get unique IDs and their positions
    unique_ids, inverse_indices = np.unique(id_column, return_inverse=True)
    n_groups = len(unique_ids)

    # Calculate group averages for all columns simultaneously
    # Shape: (n_groups, n_features)
    group_sums = np.zeros((n_groups, n_features))
    group_counts = np.bincount(inverse_indices)

    # Vectorized sum calculation
    for i in range(n_groups):
        mask = inverse_indices == i
        group_sums[i] = np.sum(y_hat[mask], axis=0)

    # Calculate averages
    group_averages = group_sums / group_counts.reshape(-1, 1)

    # Check for zero averages
    zero_avg_mask = group_averages == 0
    if np.any(zero_avg_mask):
        zero_groups, zero_cols = np.where(zero_avg_mask)
        raise ZeroDivisionError(
            f"Cannot normalize: groups {unique_ids[zero_groups]} "
            f"have zero average in columns {zero_cols}"
        )

    # Normalize: broadcast group averages back to original shape
    normalized = y_hat / group_averages[inverse_indices]

    return normalized


def calculate_metrics_per_day(ds: TrainingDataset, run=None, plot: bool = False) -> None:
    if run:
        table_cols = ["month", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)

    day_array = ds.datetime_column.dt.day().to_numpy()
    y_test = ds.y_test.to_numpy()
    id_to_metrics = list()
    id_to_ind_metrics = list()
    for day in range(1, 8):
        day_mask = day_array == day
        y_hat = ds.y_hat[day_mask]
        y = y_test[day_mask]

        test_mse, test_mae, test_rmse, test_nrmse, test_mape, rse_list = get_metrics(y, y_hat)

        mean_target_value = np.mean(y)
        id_to_ind_metrics.append(
            {"id": day, "rse": rse_list, "nrse": rse_list / mean_target_value, "avg_diff": mean_target_value})

        # mean over metrics if output length > 1
        if y.shape[1] > 1:
            test_mse, test_mae, test_rmse, test_nrmse, test_mape = get_mean_metrics(test_mse, test_mae, test_rmse,
                                                                                    test_nrmse, test_mape)

        if run:
            wandb_table.add_data(day, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
        curr_metrics = {"id": day, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                        "rmse": test_rmse, "avg_diff": mean_target_value, "n_entries": len(y)}
        id_to_metrics.append(curr_metrics)

    plot_bar_chart(id_to_metrics, "rmse", None, log_y=False, name="day")
    plot_bar_chart(id_to_metrics, "nrmse", None, log_y=False, name="day")
    plot_box_plot(id_to_ind_metrics, "rse", None, log_y=False, name="day")
    plot_box_plot(id_to_ind_metrics, "nrse", None, log_y=False, name="day")

if __name__ == '__main__':
    model_name = "tft"
    # Load predictions and targets from CSV files
    predictions_df = pl.read_csv(DATA_DIR / "predictions" / model_name / "p90_predictions.csv")
    targets_df = pl.read_csv(DATA_DIR / "predictions" / model_name / "targets.csv")

    # calculate_metrics_per_id(predictions_df, targets_df, None, False)
