# coding=utf-8
"""Custom formatting functions for Heat energy dataset.

Defines dataset specific column definitions and data transformations.
"""

import src.energy_forecast.tft.libs.utils as utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import re
import math

from src.energy_forecast.config import CONTINUOUS_FEATURES_CYCLIC, CATEGORICAL_FEATURES, FEATURE_SET_13
from src.energy_forecast.tft.data_formatters import base

GenericDataFormatter = base.GenericDataFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class HeatDataFormatter(GenericDataFormatter):
  """Defines and formats data for the heat energy consumption dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('datetime', DataTypes.DATE, InputTypes.TIME),
      ('diff', DataTypes.REAL_VALUED, InputTypes.TARGET),  # Energy consumption target
      ('tavg', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Average temperature
      ('tmin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Min temperature
      ('tmax', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Max temperature
      ('hum_avg', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Average humidity
      ('prcp', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Precipitation
      ('snow', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Snow
      ('wspd', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Wind speed
      ('pres', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Pressure
      ('tsun', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),  # Sunshine duration
      ('weekday', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),  # Day of week
      ('weekend', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),  # Weekend flag
      ('holiday', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),  # Holiday flag
      ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),  # Day of month
      ('daily_avg', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),  # Historical daily average
      ('heated_area', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),  # Building heated area
      ('anzahlwhg', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),  # Number of apartments
      ('primary_energy', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),  # Energy type
      ('typ', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),  # Building type
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

    self.energy = "all"
    self.dataset = "building"
    self.meta_features = FEATURE_SET_13
    self.features = FEATURE_SET_13 # can only be less features than meta features

  def split_data(self, df, train_percentage=0.8):
    """Splits data frame into training-validation-test data frames based on the time-based split logic.
    
    This implements the same logic as train_test_split_time_based() function from 
    src/energy_forecast/utils/train_test_val_split.py.
    
    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source dataframe to split.
      train_percentage: Percentage of data to use for training (default: 0.8)

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits using time-based approach.')

    # Convert datetime to pandas datetime if it's not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort data by id and datetime
    df = df.sort_values(by=['id', 'datetime']).reset_index(drop=True)

    # check if needed features are all in the dataset
    if not set(self.meta_features).issubset(set(df.columns)):
        raise ValueError(f"Features {self.meta_features} not in dataset")
    df = df.drop_nulls(subset=self.meta_features)  # remove null values for used features
    
    # Add row index for easier tracking
    df = df.reset_index(drop=False).rename(columns={'index': 'orig_index'})
    
    # Group by building ID and split
    train_dfs = []
    test_dfs = []
    val_dfs = []
    
    discarded_ids = []
    for building_id, b_df in df.groupby('id'):
        # Add building-specific index
        b_df = b_df.reset_index(drop=False).rename(columns={'index': 'b_idx'})
        
        # Calculate split indices
        split_idx = int(len(b_df) * train_percentage)
        split_idx_two = int(len(b_df) * (((1 - train_percentage) / 2) + train_percentage))
        
        # Split into train, validation, and test
        train_b_df = b_df[b_df['b_idx'] <= split_idx].drop(columns=['b_idx'])
        
        # Get the minimum required length for input/output sequence
        try:
            min_len = self.get_fixed_params()['num_encoder_steps'] + (
                self.get_fixed_params()['total_time_steps'] - self.get_fixed_params()['num_encoder_steps'])
        except KeyError:
            min_len = 7 + 7  # Default values if not in fixed params
        
        # Check if train split has enough data
        if len(train_b_df) <= min_len:
            discarded_ids.append(building_id)
            continue
        
        # Get validation and test splits
        test_b_df = b_df[(b_df['b_idx'] > split_idx) & (b_df['b_idx'] <= split_idx_two)].drop(columns=['b_idx'])
        if len(test_b_df) <= min_len:
            discarded_ids.append(building_id)
            continue
            
        val_b_df = b_df[b_df['b_idx'] > split_idx_two].drop(columns=['b_idx'])
        if len(val_b_df) <= min_len:
            discarded_ids.append(building_id)
            continue
            
        # Add to our collection of dataframes
        train_dfs.append(train_b_df)
        test_dfs.append(test_b_df)
        val_dfs.append(val_b_df)
    
    print(f"Removed {len(discarded_ids)} series because they were too short")
    print(f"Remaining series: {len(train_dfs)}")
    
    # Concatenate results
    train_df = pd.concat(train_dfs).drop(columns=['orig_index'])
    val_df = pd.concat(val_dfs).drop(columns=['orig_index'])
    test_df = pd.concat(test_dfs).drop(columns=['orig_index'])
    
    self.set_scalers(train_df)
    
    return (self.transform_inputs(data) for data in [train_df, val_df, test_df])

  def set_scalers(self, df, scale_mode='individual'):
    """Calibrates scalers using the data supplied.
    Implements scaling logic similar to TrainingDataset.fit_scalers().
    
    Args:
      df: Data to use to calibrate scalers.
      scale_mode: Scaling mode, either 'all' for global scaling or 'individual' for per-ID scaling.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                 column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                     column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers for input features (non-target)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET})  # Exclude target

    # Handle NaN values for feature inputs
    if real_inputs:
        data = df[real_inputs].fillna(0).values
        self._real_scalers = StandardScaler().fit(data)
    
    # Format target scalers based on scaling mode
    if scale_mode == 'all':
        # Global scaling - one scaler for all buildings
        self._target_scaler = StandardScaler().fit(
            df[[target_column]].fillna(0).values)
    
    elif scale_mode == 'individual':
        # Individual scaling - one scaler per building ID
        # Create a dictionary of scalers, one for each building ID
        self._target_scaler = {}
        
        # Get original IDs by removing any suffixes (e.g., "-1", "-2")
        original_ids = []
        for bid in self.identifiers:
            # Remove suffixes like "-1", "-2" to get the original building ID
            orig_id = re.sub(r'(-\d+)*$', '', bid)
            if orig_id not in original_ids:
                original_ids.append(orig_id)
        
        # Create a scaler for each original ID
        for orig_id in original_ids:
            # Get all data for this building (including any suffixed IDs)
            building_mask = df[id_column].apply(lambda x: x.startswith(orig_id))
            building_df = df[building_mask]
            
            if len(building_df) > 0:
                # Fit a scaler on this building's data
                self._target_scaler[orig_id] = StandardScaler().fit(
                    building_df[[target_column]].fillna(0).values)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
        # Set all to str so that we don't have mixed integer/string columns
        # Handle NaN values by converting them to 'unknown'
        srs = df[col].fillna('unknown').apply(str)
        categorical_scalers[col] = LabelEncoder().fit(
            srs.values)
        num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.
    Handles both global and individual scaling for the target variable.
    Encodes cyclic features using sine and cosine transformations.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.
    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

    # Get non-target real-valued inputs
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET})
    
    # Get categorical inputs excluding ID and TIME
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # only one-hot-encode non-binary categorical features
    categorical_inputs = list(set(categorical_inputs) & set(CATEGORICAL_FEATURES))

    # Define cyclic features that need encoding
    cyclic_features = CONTINUOUS_FEATURES_CYCLIC
    
    # Encode cyclic features with sine and cosine transforms
    encoded_columns = []
    for col in cyclic_features:
      if col in categorical_inputs:
        # For categorical cyclic features, first get numeric representation
        if col in output.columns:
          # Convert to numeric if it's a string
          if output[col].dtype == 'object':
            output[col] = pd.to_numeric(output[col], errors='coerce')
            
          # Define the period for each cyclic feature
          if col == 'weekday':
            period = 7  # 7 days in a week
          elif col == 'day_of_month':
            period = 31  # Max 31 days in a month
          
          # Create sine and cosine encodings
          output[f'{col}_sin'] = np.sin(2 * math.pi * output[col] / period)
          output[f'{col}_cos'] = np.cos(2 * math.pi * output[col] / period)
          
          # Add new columns to real_inputs list and update column definitions
          real_inputs.extend([f'{col}_sin', f'{col}_cos'])
          encoded_columns.extend([f'{col}_sin', f'{col}_cos'])
          
          # Remove original column from output to avoid double counting
          output = output.drop(columns=[col])

    # Fill missing values
    for col in real_inputs:
      if col in output.columns:  # Make sure column exists
        output[col] = output[col].fillna(0)
    
    # Format real feature inputs (excluding target)
    feature_inputs = [col for col in real_inputs if col != target_column and col in output.columns]
    if feature_inputs and self._real_scalers:
      output[feature_inputs] = self._real_scalers.transform(output[feature_inputs].values)

    # Format target based on scaler type (individual or global)
    if isinstance(self._target_scaler, dict):
      # Individual scaling per building
      for i, row in output.iterrows():
        building_id = row[id_column]
        # Get original ID by removing suffixes
        orig_id = re.sub(r'(-\d+)*$', '', building_id)
        
        if orig_id in self._target_scaler:
          # Scale this building's target with its specific scaler
          output.at[i, target_column] = self._target_scaler[orig_id].transform(
              [[row[target_column]]])[0][0]
    else:
      # Global scaling for all buildings
      output[target_column] = self._target_scaler.transform(
          output[[target_column]].values)

    # Format categorical inputs
    for col in categorical_inputs:
      if col in output.columns:  # Make sure column exists
        string_df = output[col].fillna('unknown').apply(str)
        output[col] = self._cat_scalers[col].transform(string_df)

    # Update column definitions to reflect the new cyclic encoded features
    for col in encoded_columns:
      self._column_definition.append((col, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))

    # choose desired features
    output = output[self.features]
    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.
    Handles both global and individual scaling for predictions.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    # Extract identifiers if present in the predictions
    has_identifiers = 'identifier' in predictions.columns

    if isinstance(self._target_scaler, dict):
      # Individual scaling per building
      if not has_identifiers:
        raise ValueError("Predictions must contain 'identifier' column when using individual scaling")
      
      # Process each building separately
      for i, row in output.iterrows():
        building_id = row['identifier']
        # Get original ID by removing suffixes
        orig_id = re.sub(r'(-\d+)*$', '', building_id)
        
        # Skip buildings we don't have scalers for
        if orig_id not in self._target_scaler:
          continue
          
        # Process each prediction column
        for col in output.columns:
          if col not in {'forecast_time', 'identifier'}:
            # Inverse transform this building's prediction with its specific scaler
            output.at[i, col] = self._target_scaler[orig_id].inverse_transform(
                [[row[col]]])[0][0]
    else:
      # Global scaling for all buildings
      for col in output.columns:
        if col not in {'forecast_time', 'identifier'}:
          output[col] = self._target_scaler.inverse_transform(predictions[[col]])

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 7 + 7,  # History + prediction horizon
        'num_encoder_steps': 7,    # History length
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 160,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params