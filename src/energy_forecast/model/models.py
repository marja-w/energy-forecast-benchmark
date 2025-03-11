import os
import shutil
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import wandb
from overrides import overrides
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn import tree
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk.wandb_run import Run
from pandas import DataFrame

from src.energy_forecast.config import MODELS_DIR, CATEGORICAL_FEATURES, CATEGORICAL_FEATURES_BINARY, \
    CONTINUOUS_FEATURES
import statsmodels.api as sm
from permetrics.regression import RegressionMetric


def root_mean_squared_error(y_true, y_pred):
    return keras.sqrt(keras.mean(keras.square(y_pred - y_true)))


class Model:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.name = ""
        pass

    def get_model(self):
        return self.model

    def init_wandb(self, X_train, X_val) -> tuple:
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
        return config, run

    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        pass

    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple:
        pass

    def log_eval_results(self, evaluator, run, len_y_test: int):
        # logging
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        if self.config["n_out"] > 1:
            run.log(data={"test_nrmse_ind": test_nrmse, "test_mse_ind": test_mse, "test_mae_ind": test_mae})
            logger.info(
                f"MSE Loss on test data per index: {test_mse}, MAE Loss on test data per index: {test_mae},"
                f" NRMSE on test data per index: {test_nrmse}"
            )
            test_nrmse = mean(test_nrmse)
            test_mse = mean(test_mse)
            test_mae = mean(test_mae)
        run.log(data={"test_data_length": len_y_test,
                      "test_mse": test_mse,
                      "test_mae": test_mae,
                      "test_nrmse": test_nrmse
                      })
        logger.info(
            f"MSE Loss on test data: {test_mse}, MAE Loss on test data: {test_mae}, NRMSE on test data: {test_nrmse}"
        )

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


class NNModel(Model):
    def __init__(self, config):
        super().__init__(config)
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
                return X, y
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
        X[self.cont_features] = self.scaler_X.transform(X[self.cont_features])
        y = self.scaler_y.transform(y)
        return X, y

    @overrides
    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        config, run = self.init_wandb(X_train, X_val)
        # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        logger.info(f"Training {self.name} on {X_train.shape}")
        # Compile the model
        try:
            optimizer = optimizers.Adam(clipvalue=config["clip"])
        except KeyError:
            optimizer = config["optimizer"]
        self.model.compile(optimizer=optimizer,
                           loss=config["loss"],
                           metrics=config["metrics"])
        # scaling, doesnt scale for scaler="none"
        X_train, y_train = self.scale_input_data(X_train, y_train)
        X_val, y_val = self.scale_input_data(X_val, y_val)

        # Train the model
        self.model.fit(X_train,
                       y_train,
                       epochs=config["epochs"],
                       validation_data=(X_val, y_val),
                       batch_size=config["batch_size"],
                       callbacks=[WandbMetricsLogger()])
        logger.success("Modeling training complete.")
        return self.model, run

    @overrides
    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple:
        # scale and encode input data if neccessary
        X_test, y_test = self.scale_input_data(X_test, y_test)
        # get predictions
        y_hat = self.model.predict(X_test)
        if self.scaler_X is not None:
            test_mse_scaled, test_mae_scaled = self.model.evaluate(X_test, y_test)
            run.log({"test_mse_scaled": test_mse_scaled, "test_mae_scaled": test_mae_scaled})
            # rescale predictions
            y_hat = self.scaler_y.inverse_transform(y_hat.reshape(len(y_hat), self.config["n_out"]))
        else:
            test_mse_scaled = None
            test_mae_scaled = None
        evaluator = RegressionMetric(y_test, y_hat)
        self.log_eval_results(evaluator, run, len(y_test))
        return test_mse_scaled, test_mae_scaled, evaluator.mean_squared_error(), evaluator.mean_absolute_error(), evaluator.normalized_root_mean_square_error()


class RegressionModel(Model):
    def __init__(self, config):
        super().__init__(config)

    @overrides()
    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> tuple[
        keras.Sequential, Run]:
        config, run = self.init_wandb(X_train, None)
        self.fit(X_train, y_train, run)
        return self.model, run

    def fit(self, X_train, y_train, run):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: DataFrame, y_test: DataFrame, run: Run) -> tuple:
        y_hat = self.model.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        self.log_eval_results(evaluator, run, len(y_test))
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        return test_mse, test_mae, test_nrmse


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
        y_hat = self.model.predict(X2)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat.to_numpy())
        b_nrmse = evaluator.normalized_root_mean_square_error()
        logger.info(f"Baseline NRMSE on test data: {b_nrmse}")
        if self.config["n_out"] == 1:
            run.log({"b_nrmse": b_nrmse})
        else:
            logger.info(f"Average Baseline NRMSE on test data: {mean(b_nrmse)}")
            run.log({"b_nrmse": mean(b_nrmse), "b_nrmse_ind": b_nrmse})
        return b_nrmse


class FCNModel(NNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "FCN1"
        input_shape = len(config["features"]) - 1
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(config['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(config["n_out"])  # perform regression
        ])


class DTModel(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "DT"
        self.model = tree.DecisionTreeRegressor()

    @overrides
    def fit(self, X_train, y_train, run):
        super().fit(X_train, y_train, run)
        logger.info(
            f"Fitted Decision Tree Model with depth={self.model.get_depth()} and {self.model.get_n_leaves()} leaves")
        run.log(data={"tree_depth": self.model.get_depth(), "n_leaves": self.model.get_n_leaves()})
        return run

    @overrides
    def save(self) -> Path:
        pass
