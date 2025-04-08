from pathlib import Path
import math
import os
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Union
import wandb
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
import darts
import numpy as np
from overrides import overrides
from permetrics import RegressionMetric
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run
from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer, Scaler, MissingValuesFiller
from darts.models import RNNModel as DartsRNNModel, TransformerModel, BlockRNNModel

from src.energy_forecast.config import MODELS_DIR
from src.energy_forecast.dataset import TrainingDataset
from src.energy_forecast.model.models import RNNModel
from src.energy_forecast.utils.data_processing import remove_null_series_by_id


class DartsModel(RNNModel):
    """ Models from darts library """

    def __init__(self, config):
        """ Implement the model architecture here"""
        super().__init__(config)

    @overrides
    def train_ds(self, ds: TrainingDataset) -> Run:
        config, run = self.init_wandb(ds.X_train)
        self.set_model(ds.X_train)  # because of logging, needs to be done after init_wandb
        train_df = ds.get_train_df(False).select(["datetime", "id"] + self.config["features"])
        val_df = ds.get_val_df(False).select(["datetime", "id"] + self.config["features"])
        static_features = list(set(self.config["features"]) - set(["diff"] + ds.cont_features))
        scaler_y = Scaler(ds.scaler_y, global_fit=True)  # TODO: multiple features scaling
        scaler_X = Scaler(ds.scaler_X, global_fit=True)

        # create list of darts Series objects
        train_list = darts.TimeSeries.from_group_dataframe(train_df.to_pandas(),
                                                           group_cols="id",
                                                           time_col="datetime",
                                                           value_cols=["diff"],
                                                           static_cols=static_features,
                                                           freq="D",
                                                           fill_missing_dates=True
                                                           )

        train_covariates_list = darts.TimeSeries.from_group_dataframe(train_df.to_pandas(),
                                                                      group_cols="id",
                                                                      time_col="datetime",
                                                                      value_cols=ds.cont_features,
                                                                      freq="D",
                                                                      fill_missing_dates=True
                                                                      )

        val_list = darts.TimeSeries.from_group_dataframe(val_df.to_pandas(),
                                                         group_cols="id",
                                                         time_col="datetime",
                                                         value_cols=["diff"],
                                                         static_cols=static_features,
                                                         freq="D",
                                                         fill_missing_dates=True
                                                         )

        val_covariates_list = darts.TimeSeries.from_group_dataframe(val_df.to_pandas(),
                                                                    group_cols="id",
                                                                    time_col="datetime",
                                                                    value_cols=ds.cont_features,
                                                                    freq="D",
                                                                    fill_missing_dates=True
                                                                    )

        # scaling
        train_list_scaled = scaler_y.fit_transform(train_list)
        val_list_scaled = scaler_y.transform(val_list)
        train_covariates_scaled = scaler_X.fit_transform(train_covariates_list)
        val_covariates_scaled = scaler_X.transform(val_covariates_list)
        ds.scaler_y = scaler_y
        ds.scaler_X = scaler_X

        # fill missing values using darts  # TODO maybe throw away rather than fill?
        transformer = MissingValuesFiller()
        train_list_filled = transformer.transform(train_list_scaled)
        val_list_filled = transformer.transform(val_list_scaled)
        train_covs_filled = transformer.transform(train_covariates_scaled)
        val_covs_filled = transformer.transform(val_covariates_scaled)
        assert sum(
            [np.isnan(t.values()).sum() for t in train_list_filled]) == 0  # there should be no nans in the train series

        self.model.fit(series=train_list_filled,
                       past_covariates=train_covs_filled,
                       val_series=val_list_filled,
                       val_past_covariates=val_covs_filled,
                       verbose=True
                       )  # TODO: val series
        return run

    @overrides(check_signature=False)
    def predict(self, X: list[darts.TimeSeries]) -> list[darts.TimeSeries]:
        return self.model.predict(self.config["n_out"], X)

    @overrides(check_signature=False)
    def evaluate_ds(self, ds: TrainingDataset, run: Run, log: bool = False) -> tuple:
        test_df = ds.get_test_df(scale=False).select(
            ["datetime", "id"] + self.config["features"])  # data will be scaled in evaluate by backtest function

        test_df = remove_null_series_by_id(test_df, self.config[
            "features"])  # again, remove series if there is only nans for a feature after cropping

        # create list of darts Series objects
        test_list = darts.TimeSeries.from_group_dataframe(test_df.to_pandas(),
                                                          group_cols="id",
                                                          time_col="datetime",
                                                          value_cols=self.config["features"],
                                                          freq="D",
                                                          fill_missing_dates=True
                                                          )
        # compute metrics
        metrics = self.evaluate(test_list, ds, run)
        return self.log_eval_results(metrics, len(test_df), run, log=log)

    @overrides(check_signature=False)
    def evaluate(self, n_series: list[darts.TimeSeries], ds: TrainingDataset, run: Run, log: bool = False) -> tuple:
        # fill missing values using darts  # TODO maybe throw away rather than fill?
        transformer = MissingValuesFiller()
        n_series = transformer.transform(n_series)

        # historical_forecasts = self.model.historical_forecasts(n_series,
        #     last_points_only=False,
        #     retrain=False,
        #     forecast_horizon=self.config["n_out"]
        # )

        metrics = self.model.backtest(series=n_series,
                                      # historical_forecasts=historical_forecasts,
                                      forecast_horizon=self.config["n_out"],
                                      retrain=False,
                                      data_transformers={"series": ds.scaler_y},
                                      last_points_only=False,  # retrieve all predicted values
                                      metric=[darts.metrics.metrics.rmse,
                                              darts.metrics.metrics.mse,
                                              darts.metrics.metrics.mae,
                                              darts.metrics.metrics.mase],
                                      reduction=np.mean  # put None to get individual test scores
                                      )
        metrics = np.vstack(metrics).mean(axis=0)  # average each metric over all series
        rmse = metrics[0]
        # max_v = np.vstack([s.max(axis=0).values() for s in n_series]).max()
        # min_v = np.vstack([s.min(axis=0).values() for s in n_series]).min()
        # nrmse = rmse - (max_v - min_v)  # TODO: nrmse
        self.log_eval_results(metrics, len(n_series), run, log)
        return metrics

    @overrides(check_signature=False)
    def log_eval_results(self, metrics, len_y_test: int, run, log: bool = True) -> tuple:
        rmse = metrics[0]
        mse = metrics[1]
        mae = metrics[2]
        mase = metrics[3]

        run.log(data={"test_data_length": len_y_test,
                      "test_mse": mse,
                      "test_mae": mae,
                      "test_rmse": rmse,
                      "test_mase": mase
                      })
        logger.info(f"MSE Loss on test data: {mse}")
        logger.info(f"MAE Loss on test data: {mae}")
        logger.info(f"RMSE on test data: {rmse}")
        logger.info(f"MASE on test data: {mase}")

    @overrides
    def save(self) -> Path:
        model_path = MODELS_DIR / f"{self.name}.pt"
        self.model.save(str(model_path))
        logger.success(f"Model saved to {model_path}")
        # copy to wandb run dir to fix symlink issue
        os.makedirs(os.path.join(wandb.run.dir, "models"))
        wandb_run_dir_model = os.path.join(wandb.run.dir, os.path.join("models", os.path.basename(model_path)))
        self.model.save(str(wandb_run_dir_model))
        # shutil.copy(model_path, wandb_run_dir_model)
        wandb.save(wandb_run_dir_model)
        return model_path


class DartsRNN(DartsModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RNN2"

    @overrides
    def set_model(self, X_train: np.ndarray):
        wandb_logger = WandbLogger(project=self.config["project"])
        self.model = DartsRNNModel(
            model="RNN",
            hidden_dim=20,
            n_rnn_layers=2,
            dropout=self.config["dropout"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["epochs"],
            input_chunk_length=self.config["n_in"],
            output_chunk_length=1,  # always 1 for RNN model
            output_chunk_shift=0,
            training_length=self.config["train_len"],
            pl_trainer_kwargs={"logger": [wandb_logger]}  # wandb logging
        )


class DartsBlockRNN(DartsModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "block_rnn"

    @overrides
    def set_model(self, X_train: np.ndarray):
        wandb_logger = WandbLogger(project=self.config["project"])
        self.model = BlockRNNModel(
            model="RNN",
            hidden_dim=25,
            n_rnn_layers=1,
            hidden_fc_sizes=[self.config["neurons"]],
            dropout=self.config["dropout"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["epochs"],
            input_chunk_length=self.config["n_in"],
            output_chunk_length=self.config["n_out"],  # has linear layer in end
            output_chunk_shift=0,
            activation=self.config["activation"],
            pl_trainer_kwargs={"logger": [wandb_logger]}  # wandb logging
        )


class DartsTransformer(DartsModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "transformer_default"
        self.model = TransformerModel(
            input_chunk_length=self.config["n_in"],
            output_chunk_length=self.config["n_out"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["epochs"],
            model_name=self.name,
            nr_epochs_val_period=1,  # number of epochs to wait before evaluating validation loss
            d_model=512,
            # dimensionality of the transformer architecture after embedding (default is 512 for text input)
            nhead=8,  # number of heads in the multi-head attention layer
            num_encoder_layers=3,  # number of encoder layers
            num_decoder_layers=3,  # number of decoder layers
            dim_feedforward=512,  # dimension of Feed-Forward Network model
            dropout=config["dropout"],
            activation=config["activation"],
            random_state=None,  # randomness of weight initialization
            save_checkpoints=False,
            force_reset=True,  # reset model if already exists
        )

    @overrides(check_signature=False)
    def create_time_series(self, df: pl.DataFrame) -> list[TimeSeries]:
        df = df.with_columns(pl.col("datetime").dt.cast_time_unit("ns").dt.replace_time_zone(None)).to_pandas()
        series = darts.TimeSeries.from_group_dataframe(df,
                                                       group_cols="id",
                                                       time_col="datetime",
                                                       value_cols=self.config["features"],
                                                       freq="D",
                                                       fill_missing_dates=True
                                                       )
        return series

    @overrides(check_signature=False)
    def train(self, train: list[darts.TimeSeries], val: list[darts.TimeSeries]) -> Run:
        # self.init_wandb()
        self.model.fit(
            series=train,
            val_series=val,
            # callbacks=[WandbMetricsLogger()]
        )

    @overrides
    def train_ds(self, ds: TrainingDataset) -> Run:
        # get data split either scaled or not
        train = ds.get_train_df(ds.scale).select(["datetime", "id"] + self.config["features"])
        val = ds.get_val_df(ds.scale).select(["datetime", "id"] + self.config["features"])
        self.scaled = True  # either way, scaling is done

        # create time series data from training and validation data
        train_series = self.create_time_series(train)
        logger.info(f"Training data length after time series transform: {len(train_series)}")
        val_series = self.create_time_series(val)
        logger.info(f"Validation data length after time series transform: {len(val_series)}")

        self.train(train_series, val_series)

    @overrides(check_signature=False)
    def evaluate_ds(self, ds: TrainingDataset, run: Run, log: bool = False) -> tuple:
        test_df = ds.get_test_df(ds.scale).select(["datetime", "id"] + self.config["features"])
        test_series = self.create_time_series(test_df)

        y_hat = self.predict(self.config["n_out"], test_series)
        evaluator = RegressionMetric(test_series.to_numpy(), y_hat)
        return self.log_eval_results(evaluator, run, len(test_df), log=log)
