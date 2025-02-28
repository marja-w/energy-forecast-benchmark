import wandb
from overrides import overrides
from tensorflow import keras
from tensorflow.keras import layers
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

    def train(self, X_train, y_train, X_val, y_val):
        # Setup wandb training
        config = self.config
        config["train_data_length"] = len(X_train)
        config["val_data_length"] = len(X_val)
        config["n_features"] = len(config["features"])
        run = wandb.init(project=config["project"],
                         config=config,
                         name=f"{self.name}_{config['energy']}_{config['n_features']}",
                         reinit=True)  # reinit to allow reinitialization of runs
        # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        logger.info(f"Training {self.name} on {X_train.shape}")
        # Compile the model
        self.model.compile(optimizer=config["optimizer"],
                           loss=config["loss"],
                           metrics=config["metrics"])
        # Train the model
        self.model.fit(X_train,
                       y_train,
                       epochs=config["epochs"],
                       validation_data=(X_val, y_val),
                       batch_size=config["batch_size"],
                       callbacks=[WandbMetricsLogger()])
        logger.success("Modeling training complete.")
        return self.model, run

    def evaluate(self, X_test, y_test):
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        y_hat = self.model.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        return test_loss, test_mae, evaluator.normalized_root_mean_square_error()

    def save(self):
        """
        Save model to disk as .keras file
        :return: path to saved model
        """
        model_path = MODELS_DIR / f"{self.name}.keras"
        self.model.save(model_path)
        logger.success(f"Model saved to {model_path}")
        return model_path


class LinearRegressorModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = "LR"

    def fit(self, X_train, y_train):
        X2 = sm.add_constant(X_train)
        est = sm.OLS(y_train, X2)
        self.model = est.fit()

    def evaluate(self, X_test, y_test):
        y_hat = self.model.predict(X_test)
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        return evaluator.normalized_root_mean_square_error()


class FCNModel(Model):
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
