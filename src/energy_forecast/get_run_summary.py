import ast
import io
import math

import pandas as pd
import wandb
from loguru import logger

from src.energy_forecast.config import REPORTS_DIR


def download_metrics_wandb(id_list, csv_name):
    global run
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("rausch-technology/ma-wahl-forecast")
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.id not in id_list:
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })
    csv_ = REPORTS_DIR / f"{csv_name}.csv"
    runs_df.to_csv(csv_)
    return csv_


def process_metrics_data(file_path, metric, name):
    """
    Process CSV data containing model metrics and create a DataFrame with metric_steps values.

    Args:
        file_path (str): The CSV content as a string

    Returns:
        pd.DataFrame: DataFrame with model names and metric_steps values
    """
    # Read the CSV content
    df = pd.read_csv(file_path)

    # Initialize lists to store data
    model_names = []
    mae_values_by_step = []

    baseline = None

    # Process each row
    for _, row in df.iterrows():
        try:
            # Extract model name
            model_name = row['name']

            # Parse the summary column which contains the metrics
            summary_str = row['summary']
            summary_dict = ast.literal_eval(summary_str)

            # Extract metric_steps values
            metric_steps = summary_dict.get(metric, [])

            # compute rmse from mse
            if metric == "test_mse_ind":
                metric_steps = [math.sqrt(x) for x in metric_steps]

            # Append data
            model_names.append(model_name)
            mae_values_by_step.append(metric_steps)

            if baseline is None:
                # get baseline values as own row
                model_name = "baseline"

                # Parse the summary column which contains the metrics
                summary_str = row['summary']
                summary_dict = ast.literal_eval(summary_str)

                # Extract metric_steps values
                metric_steps = summary_dict.get(metric.replace("test", "b"), [])

                # compute rmse from mse
                if metric == "test_mse_ind":
                    metric_steps = [math.sqrt(x) for x in metric_steps]

                # Append data
                model_names.append(model_name)
                mae_values_by_step.append(metric_steps)
                baseline = "done"

        except Exception as e:
            print(f"Error processing row: {e}")

    if metric == "test_mse_ind":
        metric = "test_rmse_ind"
    # Create result DataFrame
    result_data = []
    for model_name, mae_values in zip(model_names, mae_values_by_step):
        row_data = {'model': model_name}

        # Add each step's MAE value
        for i, mae in enumerate(mae_values):
            row_data[f'step_{i + 1}'] = mae

        row_data['metric'] = metric[:-4]

        result_data.append(row_data)

    result_df = pd.DataFrame(result_data)
    csv_ = REPORTS_DIR / f"{name}_metrics_{metric}.csv"
    result_df.to_csv(csv_)
    logger.info(f"Metrics processed and saved to {csv_}")

if __name__ == '__main__':
    # id_list = ["a11okqcm", "iue5852w", "jzbog7ui", "bdq9efyy"]  # 7-day forecast
    id_list = ["qrnnlsxk", "laykoa3o", "b0p27vcs", "t5timhic"]  # 24 hour forecast
    # res = "7_day"
    res = "24_hour"
    csv_name = f"best_models_{res}_forecast"
    download_path = download_metrics_wandb(id_list, csv_name)
    process_metrics_data(download_path, metric="test_mse_ind", name=csv_name)
