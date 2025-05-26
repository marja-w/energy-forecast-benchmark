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

"""Default configs for TFT experiments.

Contains the default output paths for data, serialised models and predictions
for the main experiments used in the publication.
"""

import os

import src.energy_forecast.tft.data_formatters.electricity
import src.energy_forecast.tft.data_formatters.volatility
import src.energy_forecast.tft.data_formatters.heat
import src.energy_forecast.tft.data_formatters.heat_no_building
import src.energy_forecast.tft.data_formatters.heat_diff
from src.energy_forecast.config import PROJ_ROOT
from src.energy_forecast.tft import data_formatters


class ExperimentConfig(object):
  """Defines experiment configs and paths to outputs.

  Attributes:
    root_folder: Root folder to contain all experimental outputs.
    experiment: Name of experiment to run.
    data_folder: Folder to store data for experiment.
    model_folder: Folder to store serialised models.
    results_folder: Folder to store results.
    data_csv_path: Path to primary data csv file used in experiment.
    hyperparam_iterations: Default number of random search iterations for
      experiment.
  """

  default_experiments = ['heat', 'heat_no_building', 'heat_diff', 'volatility', 'electricity', 'traffic', 'favorita']

  def __init__(self, experiment='volatility', root_folder=None):
    """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """

    if experiment not in self.default_experiments:
      raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
    if root_folder is None:
      root_folder = PROJ_ROOT
      print('Using root folder {}'.format(root_folder))

    self.root_folder = root_folder
    self.experiment = experiment
    self.data_folder = os.path.join(root_folder, 'data', "processed")
    self.model_folder = os.path.join(root_folder, 'models', experiment)
    self.results_folder = os.path.join(root_folder, 'reports', experiment)

    # Creates folders if they don't exist
    for relevant_directory in [
        self.root_folder, self.data_folder, self.model_folder,
        self.results_folder
    ]:
      if not os.path.exists(relevant_directory):
        os.makedirs(relevant_directory)

  @property
  def data_csv_path(self):
    csv_map = {
        'volatility': 'formatted_omi_vol.csv',
        'electricity': 'hourly_electricity.csv',
        'traffic': 'hourly_data.csv',
        'favorita': 'favorita_consolidated.csv',
        'heat': 'dataset_building_interpolate_daily_lag_7_7_feat.csv',
        'heat_no_building': 'dataset_building_interpolate_daily_lag_7_7_feat.csv',
        'heat_diff': 'dataset_building_interpolate_daily_lag_7_7_feat.csv'
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])

  @property
  def hyperparam_iterations(self):

    return 240 if self.experiment == 'volatility' else 60

  def make_data_formatter(self):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': data_formatters.volatility.VolatilityFormatter,
        'electricity': data_formatters.electricity.ElectricityFormatter,
        'heat': data_formatters.heat.HeatDataFormatter,
        'heat_no_building': data_formatters.heat_no_building.HeatNoBuildingDataFormatter,
        'heat_diff': data_formatters.heat_diff.HeatDiffDataFormatter
    }

    return data_formatter_class[self.experiment]()
