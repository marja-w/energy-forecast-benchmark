import wandb
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger

from src.energy_forecast.config import MODELS_DIR


class Model:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.name = ""
        pass

    def get_model(self):
        return self.model

    def train(self, X_train, y_train, X_val, y_val, config: dict):
        # Setup wandb training
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
        return self.model.evaluate(X_test, y_test)

    def save(self):
        """
        Save model to disk as .keras file
        :return: path to saved model
        """
        model_path = MODELS_DIR / f"{self.name}.keras"
        self.model.save(model_path)
        logger.success(f"Model saved to {model_path}")
        return model_path


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
