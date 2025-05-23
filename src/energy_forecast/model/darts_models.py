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
from src.energy_forecast.dataset import TrainingDataset, TimeSeriesDataset
from src.energy_forecast.model.models import RNNModel
from src.energy_forecast.plots import plot_per_step_metrics
from src.energy_forecast.utils.data_processing import remove_null_series_by_id


class DartsModel(RNNModel):
    """ Models from darts library """

    def __init__(self, config):
        """ Implement the model architecture here"""
        super().__init__(config)

    @overrides(check_signature=False)
    def train_ds(self, ds: TimeSeriesDataset) -> Run:
        config, run = self.init_wandb(ds.X_train)
        self.set_model(ds.X_train)  # because of logging, needs to be done after init_wandb
        train_list, train_covs = ds.get_train_series(ds.scale)
        val_list, val_covs = ds.get_val_series(ds.scale)

        self.model.fit(series=train_list,
                       past_covariates=train_covs,
                       val_series=val_list,
                       val_past_covariates=val_covs,
                       verbose=True,
                       dataloader_kwargs={'shuffle': True}
                       )
        return run

    @overrides(check_signature=False)
    def predict(self, X: list[darts.TimeSeries]) -> list[darts.TimeSeries]:
        return self.model.predict(self.config["n_out"], X)

    def evaluate_ds(self, ds: TimeSeriesDataset, run: Run, log: bool = False) -> tuple:
        test_list, test_covs = ds.get_test_series(ds.scale)

        # compute metrics
        metrics = self.evaluate(test_list, test_covs, ds, run)
        return self.log_eval_results(metrics, len(test_list), run, log=log)

    @overrides(check_signature=False)
    def evaluate(self, n_series: list[darts.TimeSeries], n_covs: Union[list[darts.TimeSeries], None],
                 ds: TrainingDataset, run: Run, log: bool = False) -> tuple:
        # historical_forecasts = self.model.historical_forecasts(n_series,
        #     last_points_only=False,
        #     retrain=False,
        #     forecast_horizon=self.config["n_out"]
        # )
        data_transformers = {"series": ds.scaler_y, "past_covariates": ds.scaler_X} if n_covs else {
            "series": ds.scaler_y}  # dont need scaler_X if we have no covariates
        metrics = self.model.backtest(series=n_series,
                                      past_covariates=n_covs,
                                      # historical_forecasts=historical_forecasts,
                                      forecast_horizon=self.config["n_out"],
                                      retrain=False,
                                      data_transformers=data_transformers,
                                      last_points_only=False,  # retrieve all predicted values
                                      metric=[darts.metrics.metrics.rmse,
                                              darts.metrics.metrics.mse,
                                              darts.metrics.metrics.mae,
                                              darts.metrics.metrics.mase],
                                      reduction=np.mean,  # put None to get individual test scores
                                      verbose=True
                                      )

        per_step_metrics = self.model.backtest(series=n_series,
                                               past_covariates=n_covs,
                                               # historical_forecasts=historical_forecasts,
                                               forecast_horizon=self.config["n_out"],
                                               retrain=False,
                                               data_transformers=data_transformers,
                                               last_points_only=False,  # retrieve all predicted values
                                               metric=[darts.metrics.metrics.ae,
                                                       darts.metrics.metrics.sle],
                                               reduction=np.mean,  # put None to get individual test scores
                                               verbose=True
                                               )
        metrics = np.vstack(metrics).mean(axis=0)  # average each metric over all series
        per_step_metrics = np.vstack(per_step_metrics).reshape(len(n_series), self.config["n_out"], 2).mean(
            axis=0)  # mean over all series to get error per step
        plot_per_step_metrics(per_step_metrics)
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
        # onnx model
        model_path_onnx = MODELS_DIR / f"{self.name}_{self.config['n_in']}_{self.config['n_out']}.onnx"
        self.model.to_onnx(str(model_path_onnx), export_params=True)
        logger.success(f"Model saved to {model_path_onnx}")

        os.makedirs(os.path.join(wandb.run.dir, "models"))
        wandb_run_dir_model = os.path.join(wandb.run.dir, os.path.join("models", os.path.basename(model_path_onnx)))
        self.model.to_onnx(str(wandb_run_dir_model), export_params=True)
        # shutil.copy(model_path, wandb_run_dir_model)
        wandb.save(wandb_run_dir_model)

        # pytorch model
        model_path = MODELS_DIR / f"{self.name}.pt"
        self.model.save(str(model_path))
        logger.success(f"Model saved to {model_path}")

        # copy to wandb run dir to fix symlink issue
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
    def set_model(self, input_shape: tuple):
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
        wandb_logger.watch(self.model)


class DartsBlockRNN(DartsModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "block_rnn"

    @overrides
    def set_model(self, input_shape: tuple):
        wandb_logger = WandbLogger(project=self.config["project"])
        self.model = BlockRNNModel(
            model="RNN",
            hidden_dim=25,
            n_rnn_layers=2,
            hidden_fc_sizes=[self.config["neurons"]],
            dropout=self.config["dropout"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["epochs"],
            input_chunk_length=self.config["n_in"],
            output_chunk_length=self.config["n_out"],  # has linear layer in end
            output_chunk_shift=0,
            activation=self.config["activation"],
            optimizer_kwargs={"lr": 0.001},
            pl_trainer_kwargs={"logger": [wandb_logger]}  # wandb logging
        )


class DartsLSTM(DartsModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "lstm"

    @overrides
    def set_model(self, input_shape: tuple):
        wandb_logger = WandbLogger(project=self.config["project"])
        self.model = BlockRNNModel(
            model="LSTM",
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


class TimeSeriesDataset(TrainingDataset):
    def __init__(self, config: dict):
        super().__init__(config)

    @overrides
    def handle_missing_features(self):
        """
        If we have missing features for a TimeSeriesDataset, we cant just drop them, as it would create holes in
        the time series. Therefore, linear interpolate them
        """
        self.df = remove_null_series_by_id(self.df, self.config["features"])

    @overrides
    def preprocess(self) -> tuple[pl.DataFrame, dict]:
        super().preprocess()
        return self.df, self.config

    def get_time_series_from_idxs(self, data_split: str, scale: bool = False) -> tuple[
        list[darts.TimeSeries], Union[list[darts.TimeSeries], None]]:
        """
        Get a list of darts.TimeSeries objects for the train-test-val split data. Return an additional list of time
        series if there are continuous features other than the heat consumption.
        :param data_split: train, test, val
        :param scale: whether to scale the data
        :return:
        """
        df = super().get_from_idxs(data_split, False)
        df = df.select(["datetime", "id"] + self.config["features"])
        df = remove_null_series_by_id(df, self.config[
            "features"])  # again, remove series if there is only nans for a feature after cropping

        # create list of darts Series objects
        target_series = darts.TimeSeries.from_group_dataframe(df.to_pandas(),
                                                              group_cols="id",
                                                              time_col="datetime",
                                                              value_cols=["diff"],
                                                              static_cols=self.static_features,
                                                              freq="D",
                                                              fill_missing_dates=True
                                                              )
        if self.cont_features is not None:
            covariate_list = darts.TimeSeries.from_group_dataframe(df.to_pandas(),
                                                                   group_cols="id",
                                                                   time_col="datetime",
                                                                   value_cols=self.cont_features,
                                                                   freq="D",
                                                                   fill_missing_dates=True
                                                                   )
        else:
            covariate_list = None  # otherwise no covariates

        # scaling
        if scale:
            scaler_y = Scaler(self.scaler_y, global_fit=True)
            scaler_X = Scaler(self.scaler_X, global_fit=True)

            if data_split == "train":
                target_series = scaler_y.fit_transform(target_series)
                if covariate_list: covariate_list = scaler_X.fit_transform(covariate_list)
                self.scaler_y = scaler_y
                self.scaler_X = scaler_X
            else:
                target_series = self.scaler_y.transform(target_series)
                if covariate_list: covariate_list = self.scaler_X.transform(covariate_list)

        # fill missing values using darts  # TODO maybe throw away rather than fill?
        transformer = MissingValuesFiller()
        target_series = transformer.transform(target_series)
        if covariate_list: covariate_list = transformer.transform(covariate_list)

        assert sum(
            [np.isnan(t.values()).sum() for t in target_series]) == 0  # there should be no nans in the train series

        return target_series, covariate_list

    def get_train_series(self, scale: bool = False) -> tuple[list[TimeSeries], list[TimeSeries]]:
        return self.get_time_series_from_idxs("train", scale)

    def get_test_series(self, scale: bool = False) -> tuple[list[TimeSeries], list[TimeSeries]]:
        """
        Get the data of the test split as target and covariate series, one series for each ID
        """
        return self.get_time_series_from_idxs("test", scale)

    def get_val_series(self, scale: bool = False) -> tuple[list[TimeSeries], list[TimeSeries]]:
        return self.get_time_series_from_idxs("val", scale)
