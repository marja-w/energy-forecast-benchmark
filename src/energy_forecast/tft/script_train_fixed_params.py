# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import argparse
import datetime
import datetime as dte
import os
from statistics import mean

import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from src.energy_forecast.tft.data_formatters import base
from tensorflow.python.keras.backend import get_session
from permetrics.regression import RegressionMetric

from src.energy_forecast.utils.metrics import root_mean_squared_error

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         extra_config,
         use_testing_mode=False):
    """Trains tft based on defined model params.

    Args:
      expt_name: Name of experiment
      use_gpu: Whether to run tensorflow with GPU operations
      model_folder: Folder path where models are serialized
      data_csv_path: Path to csv file containing data
      data_formatter: Dataset-specific data fromatter (see
        expt_settings.dataformatter.GenericDataFormatter)
      use_testing_mode: Uses a smaller models and data sizes for testing purposes
        only -- switch to False to use original default settings
    """

    num_repeats = 1

    if not isinstance(data_formatter, base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    # Tensorflow setup
    default_keras_session = get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Training from defined parameters for {} ***".format(expt_name))
    data_formatter.set_config(config=extra_config)

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    fixed_params["quantiles"] = extra_config["quantiles"]
    fixed_params["train_data_length"] = len(train)
    fixed_params["valid_data_length"] = len(valid)
    fixed_params["test_data_length"] = len(test)
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.inf
    for _ in range(num_repeats):

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

            tf.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())

            pre_training = dte.datetime.now()
            history = model.fit()
            after_training = dte.datetime.now()

            training_time = after_training - pre_training
            print(f"Training took {training_time}")

            df_history = pd.DataFrame({"loss": history.history["loss"], "val_loss": history.history["val_loss"]})
            df_history.to_csv(f"{model_folder}/history.csv")

            val_loss = model.evaluate()
            print(f"Validation loss right after training: {val_loss}")

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model, training_time=training_time)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        predictions = [data_formatter.format_predictions(output_map[f"p{int(x * 100)}"]) for x in params["quantiles"]]

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        if params["quantiles"] == [1.0]:
            losses = [root_mean_squared_error(extract_numerical_data(targets).to_numpy(),
                                              extract_numerical_data(predictions[0]).to_numpy())]
            evaluator = RegressionMetric(extract_numerical_data(targets).to_numpy(),
                                              extract_numerical_data(predictions[0]).to_numpy())

            test_mse = evaluator.mean_squared_error()
            test_mae = evaluator.mean_absolute_error()
            test_nrmse = evaluator.normalized_root_mean_square_error()

            losses += [test_mae, test_nrmse, test_mse]
            metric_names = ["rmse", "mae", "nrmse", "mse"]
            metrics = {metric: loss for metric, loss in zip(metric_names, losses)}
            df_metrics = pd.DataFrame(metrics)
            df_metrics.loc["mean"] = df_metrics.mean()  # add means as last row
            df_metrics.to_csv(f"{model_folder}/metrics.csv")
        else:
            losses = [utils.numpy_normalised_quantile_loss(
                extract_numerical_data(targets), extract_numerical_data(x),
                q) for q, x in zip(params["quantiles"], predictions)]

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    if len(params["quantiles"]) == 1:
        for m, v in metrics.items():
            print(f"{m} on Test Data: {v.mean()}")
    else:
        for q, x in zip(params["quantiles"], losses):
            print(f"Normalised Quantile Loss for Test Data: P{int(100 * q)}={x.mean()}")


if __name__ == "__main__":
    def get_args():
        """Gets settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="volatility",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes"

    name, output_folder, use_tensorflow_with_gpu = get_args()

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    extra_config = {
        "quantiles": [1.0],
        "num_epochs": 50,
        "early_stopping_patience": 100,
        "n_in": 7,
        "n_out": 7
    }
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder,
                                  f"fixed_{extra_config['n_in']}_{extra_config['n_out']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        extra_config=extra_config,
        use_testing_mode=False)  # Change to false to use original default params
