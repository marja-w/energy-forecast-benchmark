import datetime
import math
import os
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Union, Optional, List, Dict, Any, Tuple

import keras_hub
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import wandb
import tensorflow as tf

from keras.src.callbacks import LearningRateScheduler
from loguru import logger
from networkx.generators import trees
from overrides import overrides
from pandas import DataFrame
from permetrics.regression import RegressionMetric
from polars import DataFrame
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk.wandb_run import Run

from src.energy_forecast.config import MODELS_DIR, CONTINUOUS_FEATURES, REPORTS_DIR, MASKING_VALUE
from src.energy_forecast.dataset import TrainingDataset
from src.energy_forecast.plots import plot_predictions
from src.energy_forecast.utils.time_series import series_to_supervised


def root_mean_squared_error(y_true, y_pred):
    return keras.sqrt(keras.mean(keras.square(y_pred - y_true)))


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class Model:
    """Base class for all models with common functionality"""

    def __init__(self, config: Dict[str, Any]):
        self.model = None
        self.config = config
        self.name = ""

    def get_model(self):
        return self.model

    def init_wandb(self, X_train: DataFrame, X_val: Optional[DataFrame] = None) -> Tuple[Dict[str, Any], Run]:
        """Initialize wandb run and logging"""
        config = self.config
        config["train_data_length"] = len(X_train)
        if X_val is not None:
            config["val_data_length"] = len(X_val)
        config["n_features"] = len(config["features"])
        run = wandb.init(project=config["project"],
                         config=config,
                         name=f"{self.name}_{config['energy']}_{config['n_features']}",
                         reinit=True)  # reinit to allow reinitialization of runs
        logger.info(f"Training {self.name} on {X_train.shape}")
        return config, run

    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame,
              log: bool = False) -> Optional[Run]:
        """
        Trains a model using the training dataset and evaluates its performance on the validation dataset. This
        method is designed as an abstract method, requiring subclasses to implement custom training logic. The
        input dataset includes both features and labels for training and validation.

        :param log: whether to log to wandb
        :param X_train: Training dataset features.
        :type X_train: pandas.DataFrame
        :param y_train: Training dataset labels.
        :type y_train: pandas.DataFrame
        :param X_val: Validation dataset features.
        :type X_val: pandas.DataFrame
        :param y_val: Validation dataset labels.
        :type y_val: pandas.DataFrame
        :return: The run object associated with the training execution.
        :rtype: Run
        """
        raise NotImplementedError("Subclasses must implement this method")

    def train_ds(self, ds: TrainingDataset, log: bool = False) -> Optional[Run]:
        """Train using a dataset object"""
        return self.train(ds.X_train, ds.y_train, ds.X_val, ds.y_val, log)

    def predict(self, X: DataFrame) -> np.ndarray:
        """Make predictions with the model"""
        return self.model.predict(X)

    def evaluate_ds(self, ds: TrainingDataset, run: Run) -> Tuple[float, float, float, float]:
        """Evaluate using a dataset object"""
        return self.evaluate(ds.X_test, ds.y_test, run)

    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> Tuple[float, float, float, float]:
        """Evaluate model performance and log metrics"""
        y_hat = self.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        return self.log_eval_results(evaluator, run, len(y_test))

    def evaluate_per_cluster(self, X_test: DataFrame, y_test: DataFrame, run: Run, clusters: Dict[str, List[int]]) -> \
            Dict[str, Any]:
        """Evaluate model per cluster"""
        raise NotImplementedError("Subclasses must implement this method if they support cluster evaluation")

    def log_eval_results(self, evaluator, run: Optional[Run], len_y_test: int, log: bool = True) -> Tuple[
        float, float, float, float]:
        """
        Logs evaluation results and computes average metrics if the model supports
        multiple outputs. It calculates metrics such as Mean Squared Error (MSE),
        Mean Absolute Error (MAE), Normalized Root Mean Squared Error (NRMSE),
        and Root Mean Squared Error (RMSE). Additionally, the function logs these
        metrics to a run manager and summarizes metrics when dealing with multiple
        outputs.

        :param evaluator: Object capable of computing evaluation metrics. Should
            provide methods mean_squared_error, mean_absolute_error,
            normalized_root_mean_square_error, and root_mean_squared_error to
            retrieve corresponding metrics.
        :param run: Object responsible for logging data and metrics during
            execution.
        :param len_y_test: The length of the test dataset on which evaluation is
            performed.
        :type len_y_test: int
        :param log: Boolean flag indicating whether computed metrics should be
            logged using the provided run object.
        :type log: bool
        :return: Tuple containing computed metrics in the following order:
            test_mse (float), test_mae (float), test_nrmse (float), test_rmse (float).
        :rtype: Tuple[float, float, float, float]
        """
        # Get metrics
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        test_rmse = evaluator.root_mean_squared_error()

        # Handle multiple outputs
        if self.config["n_out"] > 1:
            if log:
                if run: run.log(data={"test_nrmse_ind": test_nrmse, "test_mse_ind": test_mse, "test_mae_ind": test_mae})
                logger.info(f"MSE Loss on test data per index: {test_mse}")
                logger.info(f"MAE Loss on test data per index: {test_mae}")
                logger.info(f"RMSE Loss on test data per index: {test_rmse}")
                logger.info(f"NRMSE Loss on test data per index: {test_nrmse}")
            test_nrmse = mean(test_nrmse)
            test_rmse = mean(test_rmse)
            test_mse = mean(test_mse)
            test_mae = mean(test_mae)

        # Log metrics
        if log:
            if run:
                run.log(data={"test_data_length": len_y_test,
                              "test_mse": test_mse,
                              "test_mae": test_mae,
                              "test_rmse": test_rmse,
                              "test_nrmse": test_nrmse
                              })
            logger.info(f"MSE Loss on test data: {test_mse}")
            logger.info(f"MAE Loss on test data: {test_mae}")
            logger.info(f"RMSE on test data: {test_rmse}")
            logger.info(f"NRMSE on test data: {test_nrmse}")

        return test_mse, test_mae, test_nrmse, test_rmse

    def save(self) -> Path:
        """Save model to disk and to wandb"""
        model_path = MODELS_DIR / f"{self.name}.keras"
        self.model.save(model_path)
        logger.success(f"Model saved to {model_path}")

        # Save to wandb
        os.makedirs(os.path.join(wandb.run.dir, "models"), exist_ok=True)
        wandb_run_dir_model = os.path.join(wandb.run.dir, os.path.join("models", os.path.basename(model_path)))
        self.model.save(wandb_run_dir_model)
        wandb.save(wandb_run_dir_model)

        return model_path

    def load_model_from_file(self, file_path: str):
        self.model = tf.keras.models.load_model(file_path)
        self.model.summary()
        logger.info(f"Successfully loaded {self.name} model from {file_path}")


class Baseline(Model):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "baseline"

    @overrides(check_signature=False)
    def evaluate(self, ds: TrainingDataset, run: Run, cluster_idxs=None, log: bool = True) -> tuple:
        target_vars = ["diff"] + [f"diff(t+{i})" for i in range(1, self.config["n_out"])]
        y_test = ds.get_test_df().select(["index", "datetime", "id"] + target_vars).to_pandas()
        if cluster_idxs is not None:
            y_test = y_test.iloc[cluster_idxs].sort_values(
                by=["id", "datetime"])  # TODO: does clustering work, are those the right indexes?
        y_test = y_test.drop(columns=["datetime", "index"])
        y_hat = y_test.groupby("id")[target_vars].shift(self.config["n_out"],
                                                        # shift by number of predictions to be made
                                                        fill_value=0)  # predictions are value of last n days, first day is zero
        y_test = y_test.drop(columns=["id"])

        # drop all rows that are equal to 0 and the corresponding rows in y_test
        idxs_to_drop = y_hat.loc[(y_hat == 0).all(axis=1)].index
        y_test = y_test.drop(idxs_to_drop)
        y_hat = y_hat.drop(idxs_to_drop)

        # compute metrics
        evaluator = RegressionMetric(y_test[target_vars].to_numpy(), y_hat.to_numpy())
        b_nrmse = evaluator.normalized_root_mean_square_error()
        b_rmse = evaluator.root_mean_squared_error()
        b_mae = evaluator.mean_absolute_error()
        b_mse = evaluator.mean_squared_error()
        if log:
            logger.info(f"Baseline MSE on test data: {b_mse}")
            logger.info(f"Baseline MAE on test data: {b_mae}")
            logger.info(f"Baseline RMSE on test data: {b_rmse}")
            logger.info(f"Baseline NRMSE on test data: {b_nrmse}")
        if self.config["n_out"] == 1:
            if log: run.log({"b_nrmse": b_nrmse, "b_rmse": b_rmse, "b_mae": b_mae, "b_mse": b_mse})
        else:
            if log:
                logger.info(f"Average Baseline MSE on test data: {mean(b_mse)}")
                logger.info(f"Average Baseline MAE on test data: {mean(b_mse)}")
                logger.info(f"Average Baseline RMSE on test data: {mean(b_rmse)}")
                logger.info(f"Average Baseline NRMSE on test data: {mean(b_nrmse)}")
                run.log({"b_nrmse": mean(b_nrmse), "b_nrmse_ind": b_nrmse,
                         "b_rmse": mean(b_rmse), "b_rmse_ind": b_rmse,
                         "b_mae": mean(b_mae), "b_mae_ind": b_mae,
                         "b_mse": mean(b_mse), "b_mse_ind": b_mse
                         })
        return b_mse, b_mae, b_nrmse, b_rmse

    @overrides(check_signature=False)
    def evaluate_per_cluster(self, ds: TrainingDataset, run: Run, clusters: dict) -> dict:
        eval_dict = dict()
        for (idx, cluster) in clusters.items():
            logger.info(f"Evaluating Cluster {idx}")
            eval_dict[f"baseline_cluster_{idx}"] = list(self.evaluate(ds, run, cluster, log=False))
        return eval_dict


class NNModel(Model):
    def __init__(self, config):
        super().__init__(config)
        # learning rate scheduler
        if config["lr_scheduler"] == "step_decay":
            self.lr_callback = LearningRateScheduler(step_decay, verbose=True)
        else:
            self.lr_callback = None
        # weight initializer
        match config["weight_initializer"]:
            case "normal":
                self.initializer = "normal"
            case "glorot":
                self.initializer = keras.initializers.GlorotNormal()
            case "zeros":
                self.initializer = keras.initializers.Zeros()
        self.activation = config["activation"]

        self.input_names = [f"{f}(t-{i})" for i, f in
                            product(range(self.config["n_in"], 0, -1), self.config["features"])]
        self.future_cov_names = [f"{f}(t+{i})" if i != 0 else f"{f}" for i, f in
                                 product(range(self.config["n_future"]),
                                         self.config["features"])]  # TODO use diff but mask diff=t
        self.target_names = ["diff"] + [f"diff(t+{i})" for i in range(1, self.config["n_out"])]
        self.scaled = False  # whether we have already scaled data

    def scale_input_data(self, X: DataFrame, y: DataFrame) -> tuple[DataFrame, DataFrame]:
        """
        Scale the data according to the scaler stored in config.
        Args:
            X: data features
            y: data labels

        Returns:
            scaled data and labels
        """
        config = self.config

        # reshape target variable for scaling
        y = y.to_numpy().reshape(len(y), config["n_out"])

        # set scaler if not set yet
        if self.scaler_X is None and self.scaler_y is None:
            if config["scaler"] == "minmax":
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
            elif config["scaler"] == "standard":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
            elif config["scaler"] == "none":
                self.scaler_X = None
                self.scaler_y = None
                return X, pd.DataFrame(y)
            else:
                raise NotImplementedError(f"Scaler {config['scaler']} not implemented")
        else:
            scaler_X = self.scaler_X
            scaler_y = self.scaler_y

        # fit scalers if we havent yet
        if len(self.cont_features) == 0:  # method was not called yet -> training data
            cont_features = list(set(config["features"]) & set(CONTINUOUS_FEATURES))
            if len(cont_features) > 0:
                self.scaler_X = scaler_X.fit(X[cont_features])
                self.cont_features = cont_features
            # target variable
            self.scaler_y = scaler_y.fit(y)

        # scale
        X_scaled = X.copy()  # dont scale base dataframe
        X_scaled[self.cont_features] = self.scaler_X.transform(X[self.cont_features])
        y_scaled = self.scaler_y.transform(y)

        return X_scaled, DataFrame(y_scaled)

    def get_train_and_val_df(self, ds: TrainingDataset) -> Tuple[pl.DataFrame, pl.DataFrame]:
        features = ["id"] + self.config["features"]
        return ds.get_train_df(ds.scale).select(features), ds.get_val_df(ds.scale).select(features)

    def transform_series_to_supervised(self, df) -> pl.DataFrame:
        try:
            lag_in, lag_out = self.config["lag_in"], self.config["lag_out"]
            assert lag_in >= self.config["n_in"] and lag_out >= self.config["n_out"]
        except KeyError:
            lag_in, lag_out = self.config["n_in"], self.config["n_out"]
        n_in, n_out = self.config["n_in"], self.config["n_out"]
        df = df.group_by("id").map_groups(lambda group: series_to_supervised(group, n_in, n_out, lag_in, lag_out))
        df = df.sort(["id"])
        return df

    def handle_future_covs(self, df: pl.DataFrame) -> pl.DataFrame:
        # drop all columns containing target variables
        return df.drop(self.target_names)

    def split_in_train_target(self, df):
        if self.config["n_future"] > 0:
            X = df[self.input_names + self.future_cov_names]
            X = self.handle_future_covs(X)
            y = df[self.target_names]
        else:
            X, y = df[self.input_names], df[self.target_names]
        return X, y

    def create_time_series_data(self, df: DataFrame) -> tuple[Any, pl.DataFrame]:
        df = self.transform_series_to_supervised(df)
        X, y = self.split_in_train_target(df)
        return X, y

    def create_time_series_data_and_id_map(self, df: pl.DataFrame) -> tuple[Any, pl.DataFrame, dict]:
        df = self.transform_series_to_supervised(df)
        X, y = self.split_in_train_target(df)
        id_to_data = self.create_id_to_data(df)
        return X, y, id_to_data

    def create_id_to_data(self, df: pl.DataFrame) -> dict:
        # per id
        id_to_data = dict()
        for b_id in df["id"].unique().to_list():
            b_df = df.filter(pl.col("id") == b_id)
            b_X, b_y = b_df[self.input_names], b_df[self.target_names]
            id_to_data[b_id] = (b_X, b_y)
        return id_to_data

    def set_model(self, input_shape: tuple) -> None:
        """
        Define model structure and set self.model parameter. Use X_train for setting input dimensions
        :return:
        """
        raise NotImplementedError("Every subclass needs to implement this method.")

    @overrides
    def train_ds(self, ds: TrainingDataset, log: bool = False) -> Run:
        # get data split either scaled or not
        train = ds.get_train_df(ds.scale).select(["id"] + self.config["features"])
        val = ds.get_val_df(ds.scale).select(["id"] + self.config["features"])

        # create time series data from training and validation data
        X_train, y_train = self.create_time_series_data(train)
        logger.info(f"Training data shape after time series transform: X {X_train.shape}, y {y_train.shape}")
        X_val, y_val = self.create_time_series_data(val)
        logger.info(f"Validation data shape after time series transform: X {X_val.shape}, y {y_val.shape}")

        self.set_model(X_train.shape)

        return self.train(X_train, y_train, X_val, y_val, log)

    @overrides
    def train(self, X_train: Union[np.ndarray, DataFrame], y_train: DataFrame, X_val: Union[np.ndarray, DataFrame],
              y_val: DataFrame, log: bool = False) -> Optional[Run]:
        if log:
            config, run = self.init_wandb(X_train, X_val)
        else:
            run = None
            config = self.config
        # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        # Compile the model
        try:
            optimizer = optimizers.Adam(clipvalue=config["clip"])
        except KeyError:
            optimizer = config["optimizer"]
        self.model.compile(optimizer=optimizer,
                           loss=config["loss"],
                           metrics=["mae"])

        train_callbacks = []
        if self.lr_callback is not None:
            train_callbacks.append(self.lr_callback)
        if log:
            train_callbacks.append(WandbMetricsLogger())

        self.model.summary()
        # Train the model
        self.model.fit(X_train,
                       y_train,
                       epochs=config["epochs"],
                       validation_data=(X_val, y_val),
                       batch_size=config["batch_size"],
                       callbacks=train_callbacks)
        logger.success("Model training complete.")
        return run

    def calculate_metrics_per_id(self, ds: TrainingDataset, run: Optional[Run], log: bool = True, plot: bool = False):
        # calculate metrics for each id
        id_to_test_series = ds.id_to_test_series
        id_to_metrics = list()
        if run:
            wandb_table = wandb.Table(columns=["id", "predictions", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"])
        for b_id, (X_scaled, y_scaled) in ds.id_to_test_series_scaled.items():
            y_hat_scaled = self.predict(X_scaled)
            if ds.scale:
                y_hat = ds.scaler_y.inverse_transform(
                    y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))  # TODO: scaler from scaler_y list
            else:
                y_hat = y_hat_scaled

            y = id_to_test_series[b_id][1].to_numpy()
            evaluator = RegressionMetric(y, y_hat)
            # Get metrics
            test_mse = evaluator.mean_squared_error()
            test_mae = evaluator.mean_absolute_error()
            test_rmse = evaluator.root_mean_squared_error()
            test_nrmse = test_rmse / len(y)
            test_mape = mean_absolute_percentage_error(y, y_hat)

            if self.config["n_out"] > 1:
                test_mape = mean(test_mape)
                test_nrmse = mean(test_nrmse)
                test_rmse = mean(test_rmse)
                test_mse = mean(test_mse)
                test_mae = mean(test_mae)

            if plot:
                plt = plot_predictions(ds, b_id, y_hat, self.config["lag_in"], self.config["n_out"], run, self.name)
                if run:
                    wandb_table.add_data(b_id, wandb.Image(plt), test_mape, test_mse, test_mae, test_nrmse, test_rmse, np.mean(y))
                    plt.close()
            if not run:
                b_metrics = {"id": b_id, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                             "rmse": test_rmse, "avg_diff": np.mean(y)}
                id_to_metrics.append(b_metrics)
        metrics_df = pl.DataFrame(id_to_metrics)
        if run:
            run.log({"building_metrics": wandb_table})
        else:
            metrics_df.write_csv(
                REPORTS_DIR / "metrics" / f"{self.name}_{self.config['n_out']}_{datetime.datetime.timestamp(datetime.datetime.now())}.csv")  # TODO: log to wandb

    @overrides(check_signature=False)
    def evaluate(self, ds: TrainingDataset, run: Optional[Run], log: bool = True, plot: bool = False) -> tuple:
        """
        Evaluate the NNModel on the test data. Log metrics to wandb.
        :param run:
        :param ds:
        :param log: whether to log metrics to wandb
        """
        # get data split either scaled or not
        test = ds.get_test_df(ds.scale).select(["id", "datetime"] + self.config["features"])
        ds.X_test_scaled, ds.y_test_scaled, ds.id_to_test_series_scaled = self.create_time_series_data_and_id_map(test)
        X_test_scaled, y_test_scaled = ds.X_test_scaled, ds.y_test_scaled
        if ds.scale:
            _, y_test, ds.id_to_test_series = self.create_time_series_data_and_id_map(
                ds.get_test_df(scale=False).select(["id"] + self.config["features"]))
        else:
            y_test = y_test_scaled
            ds.id_to_test_series = ds.id_to_test_series_scaled

        self.calculate_metrics_per_id(ds, run, log, plot)

        # get predictions
        y_hat_scaled = self.predict(X_test_scaled)
        if ds.scaler_y is not None:  # scaler_X might be None, if only diff as feature
            scaled_ev = RegressionMetric(y_test_scaled.to_numpy(), y_hat_scaled)
            test_mse_scaled = scaled_ev.mean_squared_error()
            test_mae_scaled = scaled_ev.mean_absolute_error()
            if log:
                run.log({"test_mse_scaled": test_mse_scaled, "test_mae_scaled": test_mae_scaled})
            # rescale predictions
            y_hat = ds.scaler_y.inverse_transform(
                y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))  # TODO list of scalers
        else:
            y_hat = y_hat_scaled

        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        test_mape = mean_absolute_percentage_error(y_test.to_numpy(), y_hat)
        if self.config["n_out"] > 1:
            if log: run.log({"test_mape_ind": test_mape})
            test_mape = mean(test_mape)
        if run: run.log({"test_mape": test_mape})
        return self.log_eval_results(evaluator, run, len(y_test), log)

    def evaluate_per_cluster(self, X_test: DataFrame, y_test: DataFrame, run: Run, clusters: dict) -> dict:
        """
        Evaluate the Model per cluster. Clusters are defined in clusters as mapping to indexes.
        """
        eval_dict = {"metric": ["mse", "mae", "nrmse", "rmse"]}
        for (idx, cluster) in clusters.items():
            # logger.info(f"Evaluating Cluster {idx}")
            cluster_X_test = X_test.iloc[cluster]
            cluster_y_test = y_test.iloc[cluster]
            eval_dict[f"model_cluster_{idx}"] = self.evaluate((cluster_X_test, cluster_y_test), run, log=False)
        return eval_dict


class FCNModel(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "FCN1"
        input_shape = len(config["features"]) - 1
        self.model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(config['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(config["n_out"])  # perform regression
        ])


class FCN2Model(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "FCN2"
        input_shape = len(config["features"]) - 1
        self.model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            layers.Dense(config["neurons"], activation=self.activation, kernel_initializer=self.initializer),
            layers.Dropout(config['dropout']),
            layers.Dense(config["neurons"], activation=self.activation, kernel_initializer=self.initializer),
            layers.Dense(config["n_out"], activation="linear")  # perform regression
        ])


class FCN3Model(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "FCN3"

    @overrides
    def set_model(self, input_shape: tuple) -> None:
        config = self.config
        input_shape = input_shape[1]
        self.model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            layers.Dense(config["neurons"], activation=self.activation, kernel_initializer=self.initializer),
            layers.Dropout(config['dropout']),
            layers.Dense(config["n_out"], activation="linear")  # perform regression
        ])


class FCN4Model(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "FCN4"
        input_shape = len(config["features"]) - 1
        self.model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            layers.Dense(64, activation='relu', kernel_initializer="normal"),
            layers.Dropout(config['dropout']),
            layers.Dense(config["n_out"], activation="linear", kernel_initializer="normal")  # perform regression
        ])


class RNNModel(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.target_names = ["diff"] + [f"diff(t+{i})" for i in range(1, self.config["n_out"])]
        self.scaled = False  # whether we have already scaled data

    @overrides
    def create_id_to_data(self, df) -> dict:
        # per id
        id_to_data = dict()
        for b_id in df["id"].unique().to_list():
            b_df = df.filter(pl.col("id") == b_id)
            b_X, b_y = b_df[self.input_names], b_df[self.target_names]

            # reshape input to be 3D [samples, timesteps, features] numpy array
            b_X = b_X.to_numpy().reshape((b_X.shape[0], self.config["n_in"], len(self.config["features"])))
            id_to_data[b_id] = (b_X, b_y)
        return id_to_data

    @overrides
    def handle_future_covs(self, df: pl.DataFrame) -> pl.DataFrame:
        # mask target variable columns, since we can not drop them
        df = df.to_pandas()
        df[self.target_names] = MASKING_VALUE
        return pl.DataFrame(df)

    @overrides
    def create_time_series_data(self, df: pl.DataFrame) -> tuple[Any, DataFrame]:
        X, y = super().create_time_series_data(df)
        X = X.to_numpy().reshape(
            (X.shape[0], self.config["n_in"] + self.config["n_future"], len(self.config["features"])))
        return X, y

    @overrides
    def create_time_series_data_and_id_map(self, df: pl.DataFrame) -> tuple[Any, pl.DataFrame, dict]:
        X, y, id_to_data = super().create_time_series_data_and_id_map(df)
        X = X.to_numpy().reshape(
            (X.shape[0], self.config["n_in"] + self.config["n_future"], len(self.config["features"])))
        return X, y, id_to_data


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    # To avoid division by zero, replace zeros in y_true with a very small number.
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe), axis=0) * 100
    return mape


class RNN1Model(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN1"

    @overrides
    def set_model(self, input_shape: tuple):
        input_shape = (input_shape[1], input_shape[2])
        self.model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Masking(mask_value=MASKING_VALUE),
            layers.SimpleRNN(self.config["neurons"]),
            layers.Dropout(self.config['dropout']),
            layers.Dense(self.config["n_out"], activation="linear")
        ])


class RNN3Model(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN3"

    @overrides
    def set_model(self, input_shape: tuple):
        self.model = keras.Sequential([
            layers.SimpleRNN(self.config["neurons"], input_shape=(input_shape[1], input_shape[2])),
            layers.SimpleRNN(self.config["neurons"]),
            layers.Dense(self.config["n_out"], activation="linear")
        ])


class LSTMModel(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "LSTM1"

    @overrides
    def set_model(self, input_shape: tuple):
        self.model = keras.Sequential([
            layers.LSTM(self.config["neurons"], input_shape=(input_shape[1], input_shape[2])),
            layers.Dropout(self.config['dropout']),
            layers.Dense(self.config["n_out"], activation="linear")
        ])


class TransformerModel2(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "transformer"

    @overrides
    def set_model(self, input_shape: tuple):
        # self.model = tfm.nlp.models.TransformerEncoder(
        #     num_layers=6,
        #     num_attention_heads=8,
        #     intermediate_size=2048,
        #     activation='relu',
        #     dropout_rate=self.config['dropout'],
        #     attention_dropout_rate=0.0,
        #     use_bias=False,
        #     norm_first=True,
        #     norm_epsilon=1e-06,
        #     intermediate_dropout=0.0,
        # )
        encoder = keras_hub.layers.TransformerEncoder(
            intermediate_dim=64,
            num_heads=input_shape[2],
            dropout=self.config["dropout"]
        )

        # Create a simple model containing the encoder.
        input = keras.Input(shape=(input_shape[1], input_shape[2]))
        output_layer = layers.Dense(self.config["n_out"], activation="linear")
        self.model = keras.Sequential([
            input,
            encoder,
            layers.GlobalAveragePooling1D(data_format="channels_last"),
            # reduce output tensor to a vector of features for each data point
            output_layer
        ])


class TransformerModel(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Transformer"

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=self.config["dropout"]
        )(inputs, inputs)
        x = layers.Dropout(self.config["dropout"])(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(self.config["dropout"])(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def set_model(self, input_shape: tuple):
        inputs = keras.Input(shape=(input_shape[1], input_shape[2]))
        x = inputs

        # TODO put in config
        num_transformer_blocks = 4
        mlp_units = [128]
        mlp_dropout = 0.1
        head_size = 256
        num_heads = 8
        ff_dim = 4

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(self.config["n_out"], activation="linear")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()


class RegressionModel(Model):
    def __init__(self, config):
        super().__init__(config)

    @overrides
    def train_ds(self, ds: TrainingDataset, log: bool = True) -> Optional[Run]:
        config, run = self.init_wandb(ds.X_train, None)
        self.fit(ds.X_train, ds.y_train)
        return run

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class LinearRegressorModel(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "LR"

    @overrides(check_signature=False)
    def fit(self, X_train, y_train):
        X2 = sm.add_constant(X_train)
        est = sm.OLS(y_train, X2)
        self.model = est.fit()

    @overrides(check_signature=False)
    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple:
        X2 = sm.add_constant(X_test, has_constant="add")
        return super().evaluate(X2, y_test, run)


class DTModel(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "DT"
        self.model = tree.DecisionTreeRegressor()

    @overrides(check_signature=False)
    def fit(self, X_train: DataFrame, y_train: DataFrame, run):
        if self.config["n_out"] == 1:
            super().fit(X_train, y_train)
            logger.info(
                f"Fitted Decision Tree Model with depth={self.model.get_depth()} and {self.model.get_n_leaves()} leaves")
            run.log(data={"tree_depth": self.model.get_depth(), "n_leaves": self.model.get_n_leaves()})
            return run
        else:  # need to fit multiple trees, one for each forecast
            trees = list()
            for i in range(self.config["n_out"]):
                t = tree.DecisionTreeRegressor()
                t.fit(X_train, y_train[i, :])
                trees.append(t)
            self.model = trees

    @overrides(check_signature=False)
    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple | None:
        if self.config["n_out"] == 1:
            return super().evaluate(X_test, y_test, run)
        else:
            y_hat = [t.predict(X_test) for t in trees]
            evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
            self.log_eval_results(evaluator, run, len(y_test))
            pass  # TODO

    @overrides
    def save(self) -> Path:
        pass  # TODO
