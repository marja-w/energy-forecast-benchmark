import math
import os
import shutil
from pathlib import Path
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd
import wandb
from keras.src.callbacks import LearningRateScheduler
from networkx.generators import trees
from overrides import overrides
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn import tree
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk.wandb_run import Run
from pandas import DataFrame
import polars as pl

from src.energy_forecast.config import MODELS_DIR, CATEGORICAL_FEATURES, CATEGORICAL_FEATURES_BINARY, \
    CONTINUOUS_FEATURES
import statsmodels.api as sm
from permetrics.regression import RegressionMetric

from src.energy_forecast.dataset import TrainingDataset
from src.energy_forecast.plots import plot_means


def root_mean_squared_error(y_true, y_pred):
    return keras.sqrt(keras.mean(keras.square(y_pred - y_true)))


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class Model:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.name = ""
        pass

    def get_model(self):
        return self.model

    def init_wandb(self, X_train: DataFrame, X_val: Union[DataFrame, None]) -> tuple:
        # Setup wandb training
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

    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        pass

    def predict(self, X: DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple:
        """
        Evaluate predictions results of model. Log metrics to wandb.run.
        :param X_test: Input data features
        :param y_test: Input data labels
        :param run: wandb run object
        :return: metrics, MSE, MAE, NRMSE
        """
        y_hat = self.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        return self.log_eval_results(evaluator, run, len(y_test))

    def evaluate_per_cluster(self, X_test: DataFrame, y_test: DataFrame, run: Run, clusters: dict) -> dict:
        """
        Evaluate the Model per cluster. Clusters are defined in clusters as mapping to indexes.
        """
        eval_dict = {"metric": ["mse", "mae", "nrmse", "rmse"]}
        for (idx, cluster) in clusters.items():
            # logger.info(f"Evaluating Cluster {idx}")
            cluster_X_test = X_test.iloc[cluster]
            cluster_y_test = y_test.iloc[cluster]
            eval_dict[f"model_cluster_{idx}"] = self.evaluate(cluster_X_test, cluster_y_test, run, log=False)
        return eval_dict

    def log_eval_results(self, evaluator, run, len_y_test: int, log: bool = True) -> tuple:
        # logging
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        test_rmse = evaluator.root_mean_squared_error()
        if self.config["n_out"] > 1:
            if log:
                run.log(data={"test_nrmse_ind": test_nrmse, "test_mse_ind": test_mse, "test_mae_ind": test_mae})
                logger.info(f"MSE Loss on test data per index: {test_mse}")
                logger.info(f"MAE Loss on test data per index: {test_mae}")
                logger.info(f"RMSE Loss on test data per index: {test_rmse}")
                logger.info(f"NRMSE Loss on test data per index: {test_nrmse}")
            test_nrmse = mean(test_nrmse)
            test_rmse = mean(test_rmse)
            test_mse = mean(test_mse)
            test_mae = mean(test_mae)
        if log:
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
        """
        Save model to disk as .keras file
        :return: path to saved model
        """
        model_path = MODELS_DIR / f"{self.name}.keras"
        self.model.save(model_path)
        logger.success(f"Model saved to {model_path}")
        # copy to wandb run dir to fix symlink issue
        os.makedirs(os.path.join(wandb.run.dir, "models"))
        wandb_run_dir_model = os.path.join(wandb.run.dir, os.path.join("models", os.path.basename(model_path)))
        self.model.save(wandb_run_dir_model)
        # shutil.copy(model_path, wandb_run_dir_model)
        wandb.save(wandb_run_dir_model)
        return model_path


class Baseline(Model):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "baseline"

    @overrides(check_signature=False)
    def evaluate(self, ds: TrainingDataset, run: Run, cluster_idxs=None, log: bool = True) -> tuple:
        y_test = ds.df.with_row_index().select(["datetime", "index", "id", "diff"]
                                               ).filter(pl.col("index").is_in(ds.test_idxs)
                                                        ).sort(["id", "datetime"])
        y_test = y_test.to_pandas()
        if cluster_idxs is not None:
            y_test = y_test.iloc[cluster_idxs]
        y_hat = y_test.groupby("id")["diff"].shift(1,
                                                   fill_value=0)  # predictions are value of yesterday, first day is zero
        evaluator = RegressionMetric(y_test["diff"].to_numpy(), y_hat.to_numpy())
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
                logger.info(f"Average Baseline MSE on test data: {b_mse}")
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
        self.scaler_X = None  # scaler of the input data
        self.scaler_y = None  # scaler of target variable
        self.cont_features = list()  # continuous features

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

    @overrides
    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        config, run = self.init_wandb(X_train, X_val)
        # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        # Compile the model
        try:
            optimizer = optimizers.Adam(clipvalue=config["clip"])
        except KeyError:
            optimizer = config["optimizer"]
        self.model.compile(optimizer=optimizer,
                           loss=config["loss"],
                           metrics=["mae"])
        # scaling, doesnt scale for scaler="none"
        X_train, y_train = self.scale_input_data(X_train, y_train)
        X_val, y_val = self.scale_input_data(X_val, y_val)

        if self.lr_callback is None:
            train_callbacks = [WandbMetricsLogger()]
        else:
            train_callbacks = [self.lr_callback, WandbMetricsLogger()]

        # Train the model
        self.model.fit(X_train,
                       y_train,
                       epochs=config["epochs"],
                       validation_data=(X_val, y_val),
                       batch_size=config["batch_size"],
                       callbacks=train_callbacks)
        logger.success("Modeling training complete.")
        return self.model, run

    @overrides
    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run, log: bool = True) -> tuple:
        """
        Evaluate the NNModel on the test data. Log metrics to wandb.
        :param log: whether to log metrics to wandb
        """
        # scale and encode input data if neccessary
        X_test_scaled, y_test_scaled = self.scale_input_data(X_test, y_test)
        # get predictions
        y_hat_scaled = self.predict(X_test_scaled)
        if self.scaler_X is not None:
            scaled_ev = RegressionMetric(y_test_scaled.to_numpy(), y_hat_scaled)
            test_mse_scaled = scaled_ev.mean_squared_error()
            test_mae_scaled = scaled_ev.mean_absolute_error()
            if log:
                run.log({"test_mse_scaled": test_mse_scaled, "test_mae_scaled": test_mae_scaled})
            # rescale predictions
            y_hat = self.scaler_y.inverse_transform(y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))
        else:
            y_hat = y_hat_scaled
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        return self.log_eval_results(evaluator, run, len(y_test), log)


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
        input_shape = len(config["features"]) - 1
        self.model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            layers.Dense(config["neurons"], activation=self.activation, kernel_initializer=self.initializer),
            layers.Dropout(config['dropout']),
            layers.Dense(config["n_out"], activation="linear")  # perform regression
        ])


class RNN1Model(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN"
        self.model = keras.Sequential([
            layers.SimpleRNN(128),
            layers.Dense(64, activation='relu'),
            layers.Dropout(config['dropout']),
            layers.Dense(32, activation='relu'),
            layers.Dense(config["n_out"])
        ])

    def series_to_supervised(self, df, n_in=1, n_out=1, dropnan=True):
        """
        Convert series to supervised learning
        Args:
            data:
            n_in:
            n_out:
            dropnan:

        Returns:

        """
        n_vars = 1 if type(df) is list else df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    @overrides
    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:  # TODO
        X_train = self.series_to_supervised(X_train, n_in=self.config["n_in"], n_out=self.config["n_out"])
        X_val = self.series_to_supervised(X_val, n_in=self.config["n_in"], n_out=self.config["n_out"])
        return super().train(X_train, y_train, X_val, y_val)


class RegressionModel(Model):
    def __init__(self, config):
        super().__init__(config)

    @overrides()
    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        config, run = self.init_wandb(X_train, None)
        self.fit(X_train, y_train)
        return self.model, run

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

    @overrides
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
