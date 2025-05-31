from statistics import mean
from typing import Union

import numpy as np
from permetrics import RegressionMetric


def get_metrics(y: np.array, y_hat: np.array) -> tuple:
    # Get metrics
    evaluator = RegressionMetric(y, y_hat)
    mse = evaluator.mean_squared_error()
    mae = evaluator.mean_absolute_error()
    mean_target_value = np.mean(y)
    rse_list = root_squared_error(y, y_hat)

    rmse = root_mean_squared_error(y, y_hat)
    nrmse = rmse / mean_target_value
    mape = mean_absolute_percentage_error(y, y_hat)

    return mse, mae, rmse, nrmse, mape, rse_list


def get_mean_metrics(mse, mae, rmse, nrmse, mape):
    mape = mean(mape)
    nrmse = mean(nrmse)
    rmse = mean(rmse)
    mse = mean(mse)
    mae = mean(mae)
    return mse, mae, rmse, nrmse, mape


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    # To avoid division by zero, replace zeros in y_true with a very small number.
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe), axis=0) * 100
    if mape.shape[0] > 1:
        return mape
    else:
        return mape[0]


def mean_squared_error(y_true: np.array, y_pred: np.array) -> Union[float, np.array]:
    """Calculate the Mean Squared Error between predicted and actual values.

    Args:
        y_true (np.array): Array of true/actual values
        y_pred (np.array): Array of predicted values

    Returns:
        float: Mean Squared Error value
    """
    return np.mean((y_true - y_pred) ** 2, axis=0)


def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> Union[float, np.array]:
    """Calculate the Root Mean Squared Error between predicted and actual values.

    Args:
        y_true (np.array): Array of true/actual values
        y_pred (np.array): Array of predicted values

    Returns:
        float: Root Mean Squared Error value
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if len(rmse) == 1:
        return rmse[0]
    return rmse


def squared_error(y_true: np.array, y_pred: np.array) -> np.array:
    """Calculate the Squared Error between predicted and actual values."""
    se = (y_true - y_pred) ** 2
    if y_true.shape[1] > 1:  # multiple outputs
        return np.mean(se, axis=1)
    return se


def root_squared_error(y_true: np.array, y_pred: np.array) -> np.array:
    """Calculate the Root Squared Error between predicted and actual values."""
    return np.sqrt(squared_error(y_true, y_pred))
