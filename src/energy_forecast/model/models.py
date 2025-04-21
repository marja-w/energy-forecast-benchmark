import math
import os
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Union

import keras_hub
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import wandb

from keras.src.callbacks import LearningRateScheduler
from loguru import logger
from networkx.generators import trees
from overrides import overrides
from pandas import DataFrame
from permetrics.regression import RegressionMetric
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk.wandb_run import Run

from src.energy_forecast.config import MODELS_DIR, CONTINUOUS_FEATURES
from src.energy_forecast.dataset import TrainingDataset
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
    def __init__(self, config):
        self.model = None
        self.config = config
        self.name = ""
        pass

    def get_model(self):
        return self.model

    def init_wandb(self, X_train: DataFrame, X_val: Union[DataFrame, None] = None) -> tuple[dict, Run]:
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

    def train(self, X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> Run:
        """
        Train the model.
        :return: wandb.Run object.
        """
        pass

    def train_ds(self, ds: TrainingDataset) -> Run:
        return self.train(ds.X_train, ds.y_train, ds.X_val, ds.y_val)

    def predict(self, X: DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate_ds(self, ds: TrainingDataset, run: Run) -> tuple:
        return self.evaluate(ds.X_test, ds.y_test, run)

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
        pass

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
    def train(self, X_train: Union[np.ndarray, DataFrame], y_train: DataFrame, X_val: Union[np.ndarray, DataFrame],
              y_val: DataFrame) -> Run:
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

        # scaling, doesnt scale for scaler="none"  TODO: integrate RNN training
        X_train, y_train = self.scale_input_data(X_train, y_train)
        X_val, y_val = self.scale_input_data(X_val, y_val)

        if self.lr_callback is None:
            train_callbacks = [WandbMetricsLogger()]
        else:
            train_callbacks = [self.lr_callback, WandbMetricsLogger()]

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

    def evaluate_ds(self, ds: TrainingDataset, run: Run) -> tuple:
        return self.evaluate((ds.X_test, ds.y_test), run)

    @overrides(check_signature=False)
    def evaluate(self, ds: Union[TrainingDataset, tuple[DataFrame, DataFrame]], run: Run, log: bool = True) -> tuple:
        """
        Evaluate the NNModel on the test data. Log metrics to wandb.
        :param run:
        :param ds:
        :param log: whether to log metrics to wandb
        """
        if type(ds) == TrainingDataset:
            y_test = ds.y_test
            if ds.scale:  # scaling happened in dataset
                X_test_scaled, y_test_scaled = ds.X_test_scaled, ds.y_test_scaled
        else:
            X_test, y_test = ds
            X_test_scaled, y_test_scaled = self.scale_input_data(X_test, y_test)

        # get predictions
        y_hat_scaled = self.predict(X_test_scaled)
        if self.scaler_y is not None:  # scaler_X might be None, if only diff as feature
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
        input_shape = len(config["features"]) - 1
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

    def set_model(self, X_train: np.ndarray):
        """
        Define model structure and set self.model parameter. Use X_train for setting input dimensions
        :return:
        """
        pass

    def create_time_series(self, df: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        feature_names = list(set(df.columns) - {"id"})
        test_feature = df["diff"]
        n_in = self.config["n_in"]
        n_out = self.config["n_out"]
        df_pandas = df.to_pandas()
        # df_p = df_pandas.groupby("id").apply(lambda group: series_to_supervised(group, n_in=n_in, n_out=n_out))
        df = df.group_by("id").map_groups(lambda group: series_to_supervised(group, n_in, n_out))
        df = df.sort("id")
        # assert df["diff"] == test_feature cant be the same because of removal of first n_in rows

        # split into input and outputs
        input_names = [f"{f}(t-{i})" for i, f in product(range(1, n_in + 1), feature_names)]
        future_cov_names = [f"{f}(t+{i})" if i != 0 else f"{f}(t)" for i, f in
                            product(range(n_in), list(set(feature_names) - {"diff"}))]
        X, y = df[input_names], df[self.target_names]

        # reshape input to be 3D [samples, timesteps, features] numpy array
        X = X.to_numpy().reshape((X.shape[0], n_in, len(feature_names)))
        return X, y

    @overrides
    def scale_input_data(self, X: DataFrame, y: DataFrame) -> tuple[DataFrame, DataFrame]:
        """ Override method so it doesnt scale in super().train() """
        if not self.scaled:
            self.scaled = True
            return super().scale_input_data(X, y)
        else:
            return X, y

    @overrides
    def train_ds(self, ds: TrainingDataset) -> Run:
        # get data split either scaled or not
        train = ds.get_train_df(ds.scale).select(["id"] + self.config["features"])
        val = ds.get_val_df(ds.scale).select(["id"] + self.config["features"])
        self.scaled = True  # either way, scaling is done

        # create time series data from training and validation data
        X_train, y_train = self.create_time_series(train)
        logger.info(f"Training data shape after time series transform: X {X_train.shape}, y {y_train.shape}")
        X_val, y_val = self.create_time_series(val)
        logger.info(f"Validation data shape after time series transform: X {X_val.shape}, y {y_val.shape}")

        self.set_model(X_train)  # need to set model before training

        return super().train(X_train, y_train, X_val, y_val)

    def evaluate_ds(self, ds: TrainingDataset, run: Run) -> tuple:
        test = ds.get_test_df(ds.scale).select(["id"] + self.config["features"])
        ds.X_test_scaled, ds.y_test_scaled = self.create_time_series(test)
        # update X_test and y_test for evaluation  # TODO: handle differently
        ds.X_test, ds.y_test = self.create_time_series(ds.get_test_df().select(["id"] + self.config["features"]))
        logger.info(f"Test data shape after time series transform: {ds.X_test.shape}")
        assert (ds.scaler_y.transform(ds.y_test) == ds.y_test_scaled).all()  # make sure the scaling is done right
        # for inverse transforming in evaluate
        self.scaler_X = ds.scaler_X
        self.scaler_y = ds.scaler_y
        super().evaluate(ds, run)


class RNN1Model(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN1"

    @overrides
    def set_model(self, X_train: np.ndarray):
        self.model = keras.Sequential([
            layers.SimpleRNN(self.config["neurons"], input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.Dropout(self.config['dropout']),
            layers.Dense(self.config["n_out"], activation="linear")
        ])


class RNN3Model(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN3"

    @overrides
    def set_model(self, X_train: np.ndarray):
        self.model = keras.Sequential([
            layers.SimpleRNN(self.config["neurons"], input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.SimpleRNN(self.config["neurons"]),
            layers.Dense(self.config["n_out"], activation="linear")
        ])


class TransformerModel2(RNNModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "transformer"

    @overrides
    def set_model(self, X_train: np.ndarray):
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
            num_heads=X_train.shape[2],
            dropout=self.config["dropout"]
        )

        # Create a simple model containing the encoder.
        input = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
        output_layer = layers.Dense(self.config["n_out"], activation="linear")
        self.model = keras.Sequential([
            input,
            encoder,
            layers.GlobalAveragePooling1D(data_format="channels_last"),  # reduce output tensor to a vector of features for each data point
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

    def set_model(self, X_train: np.ndarray):
        inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
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
    def train_ds(self, ds: TrainingDataset) -> Run:
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
