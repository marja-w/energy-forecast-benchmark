import numpy as np
import polars as pl
import wandb
from permetrics.regression import RegressionMetric
from statistics import mean

from src.energy_forecast.config import DATA_DIR, PROCESSED_DATA_DIR
from src.energy_forecast.plots import create_box_plot_predictions
from src.energy_forecast.utils.metrics import root_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error


def calculate_metrics_per_id(self, pred_df: pl.DataFrame, t_df: pl.DataFrame, run=None, plot: bool = False) -> None:
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
    meta_df = pl.read_csv(PROCESSED_DATA_DIR / "dataset_interpolate_daily_feat.csv")  # for retrieving heated area
    if run:
        table_cols = ["id", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        if plot:
            table_cols = ["id", "predictions", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
        wandb_table = wandb.Table(columns=table_cols)
    for b_id in t_df.get_column("id").unique().to_list():
        y_hat = pred_df.filter(pl.col("id") == b_id).drop(columns=["datetime", "id"]).to_numpy()
        y = t_df.filter(pl.col("id") == b_id).drop(columns=["datetime", "id"]).to_numpy()
        evaluator = RegressionMetric(y, y_hat)
        # Get metrics
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        mean_target_value = np.mean(y)
        heated_area = meta_df.filter(pl.col("id") == b_id)["heated_area"].first()
        rse_list = root_squared_error(y, y_hat)
        id_to_ind_metrics.append(
            {"id": b_id, "rse": rse_list, "nrse": rse_list / heated_area, "avg_diff": mean_target_value})
        test_rmse = root_mean_squared_error(y, y_hat)

        test_nrmse = test_rmse / mean_target_value
        test_mape = mean_absolute_percentage_error(y, y_hat)

        if self.config["n_out"] > 1:
            test_mape = mean(test_mape)
            test_nrmse = mean(test_nrmse)
            test_rmse = mean(test_rmse)
            test_mse = mean(test_mse)
            test_mae = mean(test_mae)

        if plot:
            # plt = plot_predictions() TODO
            pass
        if run and plot:
            # wandb_table.add_data(b_id, wandb.Image(plt), test_mape, test_mse, test_mae, test_nrmse, test_rmse,
            #                      mean_target_value)
            # plt.close()
            pass
        elif run:
            wandb_table.add_data(b_id, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                 mean_target_value)
        elif not run:
            b_metrics = {"id": b_id, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                         "rmse": test_rmse, "avg_diff": mean_target_value}
            id_to_metrics.append(b_metrics)
    create_box_plot_predictions(id_to_ind_metrics, "rse", run, self.config["n_out"], self.name, log_y=True)
    create_box_plot_predictions(id_to_ind_metrics, "nrse", run, self.config["n_out"], self.name, log_y=True)
    metrics_df = pl.DataFrame(id_to_metrics)
    if run:
        run.log({"building_metrics": wandb_table})
    else:
        metrics_df.write_csv(
            REPORTS_DIR / "metrics" / f"{self.name}_{self.config['n_out']}.csv")  # overwrites in next run
        logger.info(f"Metrics saved to {REPORTS_DIR}/metrics/{self.name}_{self.config['n_out']}.csv")


if __name__ == '__main__':
    model_name = "tft"
    # Load predictions and targets from CSV files
    predictions_df = pl.read_csv(DATA_DIR / "predictions" / model_name / "p90_predictions.csv")
    targets_df = pl.read_csv(DATA_DIR / "predictions" / model_name / "targets.csv")

    calculate_metrics_per_id(predictions_df, targets_df, None, False)
