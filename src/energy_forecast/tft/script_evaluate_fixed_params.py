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
import datetime as dte
import os
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import tensorflow.compat.v1 as tf
from permetrics import RegressionMetric
from tensorflow.python.keras.backend import get_session

import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
from src.energy_forecast.tft.data_formatters import base
from src.energy_forecast.utils.metrics import root_mean_squared_error, root_squared_error, \
    mean_absolute_percentage_error

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
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

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)
    opt_manager.load_results()  # loads params stored in models folder

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
        predictions = [data_formatter.format_predictions(output_map[f"p{int(x * 100)}"]) for x in
                       best_params["quantiles"]]

        # Evaluate and plot results
        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        def evaluate_and_plot(targets, predictions, output_folder):
            """Evaluates model performance and creates comparison plots."""
            metrics_by_id = {}
            id_to_ind_metrics = list()

            prediction_list = list()
            for identifier in targets['identifier'].unique():
                target_id = targets[targets['identifier'] == identifier]
                pred_id = predictions[0][predictions[0]['identifier'] == identifier]

                # Calculate metrics for this ID
                target_values = extract_numerical_data(target_id).to_numpy()
                pred_values = extract_numerical_data(pred_id).to_numpy()
                prediction_list.append({"id": identifier, "predictions": pred_values})

                evaluator = RegressionMetric(target_values, pred_values)

                rmse = root_mean_squared_error(target_values, pred_values)
                mse = evaluator.mean_squared_error()
                mae = evaluator.mean_absolute_error()
                nrmse = evaluator.normalized_root_mean_square_error()
                mape = mean_absolute_percentage_error(target_values, pred_values)

                if extra_config["n_out"] > 1:
                    rmse = mean(rmse)
                    mse = mean(mse)
                    mae = mean(mae)
                    nrmse = mean(nrmse)
                    mape = mean(mape)

                metrics_by_id[identifier] = {
                    'rmse': rmse,
                    'mse': mse,
                    'mae': mae,
                    'nrmse': nrmse,
                    'mape': mape,
                    'mean_consumption': target_values.mean()
                }

                rse_list = root_squared_error(target_values, pred_values)
                id_to_ind_metrics.append({"id": identifier, "rse": rse_list, "rse_n": rse_list / target_values.mean(),
                                          "avg_diff": target_values.mean()})

                # Create plot for this ID
                forecast_length = pred_values.shape[1]
                date_range = pd.date_range(
                    start=pred_id["forecast_time"].min(),
                    end=pred_id["forecast_time"].max() + pd.Timedelta(days=forecast_length),
                    freq='D'
                )

                params = {'axes.labelsize': 12, 'axes.titlesize': 14, 'font.size': 12, 'legend.fontsize': 12,
                          'xtick.labelsize': 10, 'ytick.labelsize': 12}
                matplotlib.rcParams.update(params)

                plt.figure(figsize=(12, 6))
                plt.plot(date_range[:len(target_values)], target_values[:, 0], label='Actual', marker='o')
                if forecast_length > 1:
                    for i in range(len(pred_values)):
                        plt.plot(date_range[i:forecast_length + i], pred_values[i, :], label=f'Forecast {i + 1}',
                                 marker='x')
                else:
                    plt.plot(date_range[:len(pred_values)], pred_values, label='Forecast', marker='x')
                # plt.title(f'Forecast vs Actual for ID: {identifier}')

                # SMALL_SIZE = 12
                # MEDIUM_SIZE = 14
                # BIGGER_SIZE = 16
                #
                # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
                # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
                # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
                # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
                # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
                # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
                # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

                plt.xlabel("Timestamp")
                plt.ylabel("Energy Consumption (kWh)")

                plt.legend()
                plt.grid(True)
                output_folder_path = f'{output_folder}/predictions/forecast_id_{identifier}.png'
                os.makedirs(os.path.dirname(output_folder_path), exist_ok=True)
                plt.savefig(output_folder_path, dpi=300)
                plt.close()

            df_predictions = pd.DataFrame(prediction_list)
            df_predictions.to_csv(f"{model_folder}/predictions.csv", index=False)

            def create_box_plot_predictions(id_to_metrics: list, metric_to_plot: str,
                                            log_y: bool = False, model_folder: str = ""):
                df = pd.DataFrame(id_to_metrics).explode(metric_to_plot)
                df[metric_to_plot] = df[metric_to_plot].astype(float)
                df = df.sort_values("avg_diff")

                fig = px.box(df, x="id", y=metric_to_plot, log_y=log_y, custom_data=["avg_diff"], width=1200, height=400)
                metric_to_plot = "RSE" if metric_to_plot == "rse" else "nRSE"
                fig.update_traces(
                    hovertemplate="<br>".join([
                        "ID: %{x}",
                        metric_to_plot + ": %{y}",
                        "avg_diff: %{customdata[0]}"
                    ])
                )
                fig.update_xaxes(showticklabels=False)
                fig.update_layout(
                    yaxis=dict(
                        title=dict(
                            text=f"{metric_to_plot} {'(log)' if log_y else ''}"
                        )
                    ),
                    xaxis=dict(
                        title=None
                    ),
                    font_size=16,
                    margin=dict(l=20, r=0, t=0, b=0)
                )
                # fig.update_yaxes(automargin=True)
                print("Writing box plot...")
                output_name = f"{model_folder}/boxplot_{metric_to_plot}_t"
                fig.write_image(f"{output_name}.png", scale=2)
                print(f"Wrote box plot to {output_name}")
                fig.write_html(f"{output_name}.html")

            create_box_plot_predictions(id_to_ind_metrics, "rse", log_y=True, model_folder=output_folder)
            create_box_plot_predictions(id_to_ind_metrics, "rse_n", log_y=True, model_folder=output_folder)
            # create_box_plot_predictions(id_to_ind_metrics, ["rse", "rse_n"], log_y=True, model_folder=output_folder)

            # Save metrics by ID
            metrics_df = pd.DataFrame.from_dict(metrics_by_id, orient='index')
            metrics_df.to_csv(f'{output_folder}/metrics_by_id.csv')

            return metrics_df

        metrics_by_id = evaluate_and_plot(targets, predictions, model_folder)

        if best_params["quantiles"] == [1.0]:
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
            df_metrics = pd.DataFrame(metrics, index=range(extra_config["n_out"]))
            df_metrics.loc["mean"] = df_metrics.mean()  # add means as last row
            df_metrics.to_csv(f"{model_folder}/metrics.csv")
        else:
            losses = [utils.numpy_normalised_quantile_loss(
                extract_numerical_data(targets), extract_numerical_data(x),
                q) for q, x in zip(best_params["quantiles"], predictions)]

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    if len(best_params["quantiles"]) == 1:
        for m, v in metrics.items():
            print(f"{m} on Test Data: {v.mean()}")
    else:
        for q, x in zip(best_params["quantiles"], losses):
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
        "n_out": 1
    }
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder, "fixed_7_1_20250526_160841"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False)  # Change to false to use original default params
