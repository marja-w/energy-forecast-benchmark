import os
import shutil

import wandb
from overrides import overrides
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import tree
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger

from src.energy_forecast.config import MODELS_DIR
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

    def train(self, X_train, y_train, X_val, y_val):
        pass

    def evaluate(self, X_test, y_test, run):
        pass

    def save(self):
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

    def preprocess_input_data(self, X_train, X_val, y_train, y_val):
        config = self.config
        # scale the input data
        if config["scaler"] == "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif config["scaler"] == "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        else:
            raise NotImplementedError(f"Scaler {config['scaler']} not implemented")
        self.scaler_X = scaler_X.fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        # scale target variable
        y_train_reshape = y_train.to_numpy().reshape(len(y_train), 1)
        y_val_reshape = y_val.to_numpy().reshape(len(y_val), 1)
        self.scaler_y = scaler_y.fit(y_train_reshape)
        y_train_scaled = scaler_y.transform(y_train_reshape)
        y_val_scaled = scaler_y.transform(y_val_reshape)
        return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled

    @overrides
    def train(self, X_train, y_train, X_val, y_val):
        config, run = self.init_wandb(X_train, X_val)
        # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        logger.info(f"Training {self.name} on {X_train.shape}")
        # Compile the model
        self.model.compile(optimizer=config["optimizer"],
                           loss=config["loss"],
                           metrics=config["metrics"])
        # scaling
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = self.preprocess_input_data(X_train, X_val,
                                                                                                y_train, y_val)
        # Train the model
        self.model.fit(X_train_scaled,
                       y_train_scaled,
                       epochs=config["epochs"],
                       validation_data=(X_val_scaled, y_val_scaled),
                       batch_size=config["batch_size"],
                       callbacks=[WandbMetricsLogger()])
        logger.success("Modeling training complete.")
        return self.model, run

    def evaluate(self, X_test, y_test, run):
        # scale input data
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test.to_numpy().reshape(len(y_test), 1))
        # get predictions
        y_hat_scaled = self.model.predict(X_test_scaled)
        test_mse_scaled, test_mae_scaled = self.model.evaluate(X_test_scaled, y_test_scaled)
        # rescale predictions
        y_hat = self.scaler_y.inverse_transform(y_hat_scaled.reshape(len(y_hat_scaled), 1))
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        # logging
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        run.log(data={"test_data_length": len(y_test),
                      "test_mse_scaled": test_mse_scaled,
                      "test_mae_scaled": test_mae_scaled,
                      "test_mse": test_mse,
                      "test_mae": test_mae,
                      "test_nrmse": test_nrmse
                      })
        logger.info(
            f"MSE Loss on test data: {test_mse}, MAE Loss on test data: {test_mae}, NRMSE on test data: {test_nrmse}"
        )
        return test_mse_scaled, test_mae_scaled, evaluator.mean_squared_error(), evaluator.mean_absolute_error(), evaluator.normalized_root_mean_square_error()


class RegressionModel(Model):
    def __init__(self, config):
        super().__init__(config)

    @overrides()
    def train(self, X_train, y_train, X_val, y_val):
        config, run = self.init_wandb(X_train, None)
        self.fit(X_train, y_train, run)
        return self.model, run

    def fit(self, X_train, y_train, run):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, run):
        y_hat = self.model.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        test_nrmse = evaluator.normalized_root_mean_square_error()
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        run.log(data={"test_data_length": len(y_test), "test_nrmse": test_nrmse, "test_mse": test_mse,
                      "test_mae": test_mae})
        logger.info(
            f"MSE Loss on test data: {test_mse}, MAE Loss on test data: {test_mae}, NRMSE on test data: {test_nrmse}"
        )
        return test_mse, test_mae, test_nrmse


class LinearRegressorModel(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "LR"

    def fit(self, X_train, y_train, run):
        X2 = sm.add_constant(X_train)
        est = sm.OLS(y_train, X2)
        self.model = est.fit()

    @overrides
    def evaluate(self, X_test, y_test, run):
        X2 = sm.add_constant(X_test)
        y_hat = self.model.predict(X2)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat.to_numpy())
        b_nrmse = evaluator.normalized_root_mean_square_error()
        logger.info(f"Baseline NRMSE on test data: {b_nrmse}")
        run.log({"b_nrmse": b_nrmse})
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
            layers.Dense(1)  # perform regression
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
    def save(self):
        pass
