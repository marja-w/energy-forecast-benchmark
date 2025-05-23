import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from permetrics.regression import RegressionMetric
from tqdm import tqdm
import jsonlines

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

from src.energy_forecast.config import REFERENCES_DIR, MODELS_DIR, REPORTS_DIR
from src.energy_forecast.dataset import TrainingDataset, TrainDataset90, TrainDatasetBuilding
from src.energy_forecast.model.train import prepare_dataset, get_data
from src.energy_forecast.utils.metrics import mean_absolute_percentage_error, root_mean_squared_error, root_squared_error
from src.energy_forecast.utils.util import store_df_wandb
from src.energy_forecast.plots import plot_predictions, create_box_plot_predictions


class xLSTMModel:
    """PyTorch model implementation of the xLSTM model"""

    def __init__(self, config):
        self.config = config
        self.name = "xLSTM"
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.criterion = None
        self.scaler_X = None
        self.scaler_y = None
        self.cont_features = None
        
        # Set up model parameters based on config
        self.input_names = [f"{f}(t-{i})" for i in range(self.config["n_in"], 0, -1) 
                            for f in self.config["features"]]
        self.future_cov_names = [f"{f}(t+{i})" if i != 0 else f"{f}" for i in range(self.config["n_future"])
                                for f in self.config["features"]]
        self.target_names = ["diff"] + [f"diff(t+{i})" for i in range(1, self.config["n_out"])]

    def set_model(self, input_shape):
        """Initialize the xLSTM model"""
        input_dim = input_shape[1]
        
        # Configure xLSTM model based on config parameters
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, 
                    qkv_proj_blocksize=4, 
                    num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.config.get("context_length", 256),
            num_blocks=self.config.get("num_blocks", 7),
            embedding_dim=self.config.get("embedding_dim", 128),
            slstm_at=[1],
        )
        
        # Initialize xLSTM model
        self.model = xLSTMBlockStack(cfg)
        
        # Add final prediction layer to output the right number of values
        self.model = nn.Sequential(  # TODO: directly set output dimensions in xLSTMBlockStackConfig?
            self.model,
            nn.Linear(cfg.embedding_dim, self.config["n_out"])
        )
        
        # Move model to the appropriate device
        self.model = self.model.to(self.device)
        
        # Set optimizer and loss function
        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.get("learning_rate", 0.001),
                weight_decay=self.config.get("weight_decay", 0)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.config.get("learning_rate", 0.01)
            )
            
        if self.config["loss"] == "mean_squared_error":
            self.criterion = nn.MSELoss()
        elif self.config["loss"] == "mean_absolute_error":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()  # Default to MSE

    def init_wandb(self, X_train, X_val=None):
        """Initialize wandb run and logging"""
        config = self.config
        config["train_data_length"] = len(X_train)
        if X_val is not None:
            config["val_data_length"] = len(X_val)
        config["n_features"] = len(config["features"])
        run = wandb.init(
            project=config["project"],
            config=config,
            name=f"{self.name}_{config['energy']}_{config['n_features']}",
            reinit=True
        )  # reinit to allow reinitialization of runs
        logger.info(f"Training {self.name} on {X_train.shape}")
        return config, run

    def transform_series_to_supervised(self, df):
        """Transform time series data into supervised learning format"""
        from src.energy_forecast.utils.time_series import series_to_supervised
        
        try:
            lag_in, lag_out = self.config["lag_in"], self.config["lag_out"]
            assert lag_in >= self.config["n_in"] and lag_out >= self.config["n_out"]
        except KeyError:
            lag_in, lag_out = self.config["n_in"], self.config["n_out"]
            
        n_in, n_out = self.config["n_in"], self.config["n_out"]
        
        if not lag_out >= self.config["n_future"]:
            raise ValueError("n_future cannot be larger than lag out")
            
        df = df.group_by("id").map_groups(lambda group: series_to_supervised(group, n_in, n_out, lag_in, lag_out))
        df = df.sort(["id"])
        return df

    def handle_future_covs(self, df):
        """Handle future covariates for the model input"""
        from src.energy_forecast.config import MASKING_VALUE
        
        # mask target variable columns, since we can not drop them
        df = df.to_pandas()
        lag_target_names = ["diff"] + [f"diff(t+{i})" for i in range(1, self.config["lag_out"])]
        lag_target_names = list(set(df.columns).intersection(lag_target_names))
        df[lag_target_names] = MASKING_VALUE
        return pl.DataFrame(df)

    def split_in_feature_target(self, df):
        """Split the dataframe into features and targets"""
        if self.config["n_future"] > 0:
            X = df[self.input_names + self.future_cov_names]
            X = self.handle_future_covs(X)
            y = df[self.target_names]
        else:
            X, y = df[self.input_names], df[self.target_names]
        return X, y

    def create_time_series_data(self, df):
        """Create time series data from dataframe"""
        df = self.transform_series_to_supervised(df)
        X, y = self.split_in_feature_target(df)
        return X, y

    def create_time_series_data_and_id_map(self, df):
        """Create time series data and ID map for evaluation"""
        df = self.transform_series_to_supervised(df)
        X, y = self.split_in_feature_target(df)
        id_to_data = self.create_id_to_data(df)
        return X, y, id_to_data, df["id"]

    def create_id_to_data(self, df):
        """Create a mapping from IDs to data"""
        id_to_data = {}
        for b_id in df["id"].unique().to_list():
            b_df = df.filter(pl.col("id") == b_id)
            date_c = b_df["datetime"]
            b_X, b_y = self.split_in_feature_target(b_df)
            id_to_data[b_id] = (b_X, b_y, date_c)
        return id_to_data

    def train(self, X_train, y_train, X_val, y_val, log=False):
        """Train the model"""
        # Initialize wandb if logging is enabled
        if log:
            config, run = self.init_wandb(X_train, X_val)
        else:
            run = None
            config = self.config
            
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).to(self.device)
        
        # Create DataLoader
        batch_size = self.config["batch_size"]
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        print(self.model)
        # Training loop
        epochs = self.config["epochs"]
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                mae = torch.nn.functional.l1_loss(outputs, targets)
                train_mae += mae.item() * inputs.size(0)
                
            train_loss = train_loss / len(train_loader.dataset)
            train_mae = train_mae / len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    mae = torch.nn.functional.l1_loss(outputs, targets)
                    val_mae += mae.item() * inputs.size(0)
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_mae = val_mae / len(val_loader.dataset)
            
            # Log metrics
            if log:
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae
                })
                
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
        logger.success("Model training complete.")
        return run

    def train_ds(self, ds, log=False):
        """Train using a dataset object"""
        # Get data split either scaled or not
        train = ds.get_train_df(ds.scale).select(["id"] + self.config["features"])
        val = ds.get_val_df(ds.scale).select(["id"] + self.config["features"])
        
        # Create time series data from training and validation data
        X_train, y_train = self.create_time_series_data(train)
        logger.info(f"Training data shape after time series transform: X {X_train.shape}, y {y_train.shape}")
        X_val, y_val = self.create_time_series_data(val)
        logger.info(f"Validation data shape after time series transform: X {X_val.shape}, y {y_val.shape}")
        
        # Set up the model
        self.set_model(X_train.shape)
        
        return self.train(X_train, y_train, X_val, y_val, log)

    def predict(self, X):
        """Make predictions with the model"""
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

    def evaluate(self, ds, run, log=True, plot=False):
        """Evaluate model performance and log metrics"""
        # Get data split either scaled or not
        test = ds.get_test_df(ds.scale).select(["id", "datetime"] + self.config["features"])
        ds.X_test_scaled, ds.y_test_scaled, ds.id_to_test_series_scaled, id_column = self.create_time_series_data_and_id_map(test)
        X_test_scaled, y_test_scaled = ds.X_test_scaled, ds.y_test_scaled
        
        if ds.scale:
            _, y_test, ds.id_to_test_series, _ = self.create_time_series_data_and_id_map(
                ds.get_test_df(scale=False).select(["id", "datetime"] + self.config["features"]))
        else:
            y_test = y_test_scaled
            ds.id_to_test_series = ds.id_to_test_series_scaled
            
        # Calculate metrics per ID
        self.calculate_metrics_per_id(ds, run, plot)
        
        # Get predictions
        y_hat_scaled = self.predict(X_test_scaled)
        
        if ds.scaler_y is not None:
            scaled_ev = RegressionMetric(y_test_scaled.to_numpy(), y_hat_scaled)
            test_mse_scaled = scaled_ev.mean_squared_error()
            test_mae_scaled = scaled_ev.mean_absolute_error()
            if run:
                run.log({"test_mse_scaled": test_mse_scaled, "test_mae_scaled": test_mae_scaled})
            # Rescale predictions
            y_hat = ds.rescale_predictions(y_hat_scaled, id_column)
        else:
            y_hat = y_hat_scaled
            
        evaluator = RegressionMetric(y_test.to_numpy(), y_hat)
        test_mape = mean_absolute_percentage_error(y_test.to_numpy(), y_hat)
        
        if self.config["n_out"] > 1:
            if run: run.log({"test_mape_ind": test_mape})
            if log: logger.info(f"MAPE on individual test data: {test_mape}")
            test_mape = sum(test_mape) / len(test_mape)
            
        if run: run.log({"test_mape": test_mape})
        if log: logger.info(f"MAPE on test data: {test_mape}")
        
        return self.log_eval_results(evaluator, run, len(y_test), log)
        
    def log_eval_results(self, evaluator, run, len_y_test, log=True):
        """Log evaluation results"""
        # Get metrics
        test_mse = evaluator.mean_squared_error()
        test_mae = evaluator.mean_absolute_error()
        test_nrmse = evaluator.normalized_root_mean_square_error()
        test_rmse = evaluator.root_mean_squared_error()
        
        # Handle multiple outputs
        if self.config["n_out"] > 1:
            if run: run.log(data={"test_nrmse_ind": test_nrmse, "test_mse_ind": test_mse, "test_mae_ind": test_mae})
            if log:
                logger.info(f"MSE Loss on test data per index: {test_mse}")
                logger.info(f"MAE Loss on test data per index: {test_mae}")
                logger.info(f"RMSE Loss on test data per index: {test_rmse}")
                logger.info(f"NRMSE Loss on test data per index: {test_nrmse}")
            test_nrmse = sum(test_nrmse) / len(test_nrmse)
            test_rmse = sum(test_rmse) / len(test_rmse)
            test_mse = sum(test_mse) / len(test_mse)
            test_mae = sum(test_mae) / len(test_mae)
            
        # Log metrics
        if run:
            run.log(data={"test_data_length": len_y_test,
                         "test_mse": test_mse,
                         "test_mae": test_mae,
                         "test_rmse": test_rmse,
                         "test_nrmse": test_nrmse
                         })
        if log:
            logger.info(f"MSE Loss on test data: {test_mse}")
            logger.info(f"MAE Loss on test data: {test_mae}")
            logger.info(f"RMSE on test data: {test_rmse}")
            logger.info(f"NRMSE on test data: {test_nrmse}")
            
        return test_mse, test_mae, test_nrmse, test_rmse
    
    def calculate_metrics_per_id(self, ds, run, plot=False):
        """Calculate metrics for each building ID"""
        import re
        from statistics import mean
        
        id_to_test_series = ds.id_to_test_series
        id_to_metrics = []
        id_to_ind_metrics = []
        
        if run:
            table_cols = ["id", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
            if plot:
                table_cols = ["id", "predictions", "mape", "mse", "mae", "nrmse", "rmse", "avg_diff"]
            wandb_table = wandb.Table(columns=table_cols)
            
        for b_id, (X_scaled, y_scaled, dates) in ds.id_to_test_series_scaled.items():
            y_hat_scaled = self.predict(X_scaled)
            
            if ds.scale:
                if self.config["scale_mode"] == "individual":
                    scaler_key = re.sub(r'(-\d+)*$', '', b_id)  # get base id without suffixes
                    y_hat = ds.scaler_y[scaler_key].inverse_transform(
                        y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))
                else:
                    y_hat = ds.scaler_y.inverse_transform(
                        y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))
            else:
                y_hat = y_hat_scaled
                
            y = id_to_test_series[b_id][1].to_numpy()
            evaluator = RegressionMetric(y, y_hat)
            
            # Get metrics
            test_mse = evaluator.mean_squared_error()
            test_mae = evaluator.mean_absolute_error()
            mean_target_value = np.mean(y)
            heated_area = ds.get_heated_area_by_id(b_id)
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
                plt = plot_predictions(ds, y, b_id, y_hat, dates, self.config["lag_in"], self.config["n_out"],
                                      self.config["lag_out"], run, self.name)
                
            if run and plot:
                wandb_table.add_data(b_id, wandb.Image(plt), test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                    mean_target_value)
                plt.close()
            elif run:
                wandb_table.add_data(b_id, test_mape, test_mse, test_mae, test_nrmse, test_rmse,
                                    mean_target_value)
            elif not run:
                b_metrics = {"id": b_id, "mape": test_mape, "mse": test_mse, "mae": test_mae, "nrmse": test_nrmse,
                            "rmse": test_rmse, "avg_diff": mean_target_value}
                id_to_metrics.append(b_metrics)
                
        create_box_plot_predictions(id_to_ind_metrics, "rse", run, log_y=True)
        create_box_plot_predictions(id_to_ind_metrics, "nrse", run, log_y=True)
        
        metrics_df = pl.DataFrame(id_to_metrics)
        if run:
            run.log({"building_metrics": wandb_table})
        else:
            metrics_df.write_csv(
                REPORTS_DIR / "metrics" / f"{self.name}_{self.config['n_out']}.csv")
            logger.info(f"Metrics saved to {REPORTS_DIR}/metrics/{self.name}_{self.config['n_out']}.csv")
    
    def save(self):
        """Save model to disk and to wandb"""
        model_path = MODELS_DIR / f"{self.name}.pt"
        
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, model_path)
        
        logger.success(f"Model saved to {model_path}")
        
        # Save to wandb
        if wandb.run:
            os.makedirs(os.path.join(wandb.run.dir, "models"), exist_ok=True)
            wandb_run_dir_model = os.path.join(wandb.run.dir, os.path.join("models", os.path.basename(model_path)))
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
            }, wandb_run_dir_model)
            wandb.save(wandb_run_dir_model)
            
        return model_path
    
    def load_model_from_file(self, file_path):
        """Load model from a file"""
        checkpoint = torch.load(file_path)
        
        # Initialize model with the saved config
        self.config = checkpoint['config']
        self.set_model((None, len(self.config["features"]) * self.config["n_in"]))
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Successfully loaded {self.name} model from {file_path}")


def get_model(config):
    """Factory function to get the appropriate model"""
    return xLSTMModel(config)


def train(run_config):
    """Main training function"""
    ds, run_config = prepare_dataset(run_config)
    
    # Get model
    m = get_model(run_config)
    
    # Train
    run = m.train_ds(ds, log=run_config["log"])
    
    # Evaluate the model
    m.evaluate(ds, run, log=run_config["log"], plot=run_config["plot"])
    
    # Save model
    m.save()
    
    return run


if __name__ == '__main__':
    configs_path = REFERENCES_DIR / "configs.jsonl"
    
    config = {
        "project": "ma-wahl-forecast",
        "log": True,  # whether to log to wandb
        "plot": False,  # whether to plot predictions
        "energy": "all",
        "res": "daily",
        "interpolate": 1,
        "dataset": "building",  # building, meta, missing_data_90
        "model": "xLSTM",
        "lag_in": 7,
        "lag_out": 7,
        "n_in": 7,
        "n_out": 7,
        "n_future": 0,
        "scaler": "standard",
        "scale_mode": "individual",  # all, individual
        "feature_code": 14,
        "train_test_split_method": "time",
        "epochs": 1,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "loss": "mean_squared_error",
        "metrics": ["mae"],
        "batch_size": 32,
        "dropout": 0.1,
        "context_length": 256,
        "num_blocks": 7,
        "embedding_dim": 128,
    }
    
    # Run training
    wandb_run = train(config)
    if wandb_run: wandb_run.finish()