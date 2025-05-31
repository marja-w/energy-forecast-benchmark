import datetime
import math
import os
import re

import numpy as np
import pandas as pd
import polars as pl
import wandb
from loguru import logger
from overrides import overrides
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

from src.energy_forecast.config import DATA_DIR, PROCESSED_DATA_DIR, CATEGORICAL_FEATURES, FEATURES, \
    FEATURES_DIR, META_DIR, INTERIM_DATA_DIR, N_CLUSTER, REPORTS_DIR, CONTINUOUS_FEATURES_CYCLIC, CONTINUOUS_FEATURES, \
    RAW_DATA_DIR, FEATURE_SET_8, FEATURE_SET_10, N_LAG, FEATURE_SETS, FEATURES_HOURLY, FEATURES_DAILY, \
    THRESHOLD_FLAT_LINES_DAILY, THRESHOLD_FLAT_LINES_HOURLY, MIN_GAP_SIZE_DAILY, MIN_GAP_SIZE_HOURLY, FEATURE_SET_15
from src.energy_forecast.data_processing.data_loader import DHDataLoader, KinergyDataLoader
from src.energy_forecast.plots import plot_missing_dates_per_building, plot_clusters
from src.energy_forecast.utils.cluster import hierarchical_clustering_on_meta_data
from src.energy_forecast.utils.data_processing import remove_neg_diff_vals, filter_connection_errors_by_id, \
    filter_outliers_by_id, filter_flat_lines_by_id, split_series_by_id_list, interpolate_values_by_id, \
    remove_positive_jumps, split_series_by_id, remove_null_series_by_id
from src.energy_forecast.utils.time_series import series_to_supervised
from src.energy_forecast.utils.util import get_missing_dates


class Dataset:
    def __init__(self, res: str = "daily"):
        self.res = res
        if res == "daily":
            self.data_sources = ["kinergy", "dh", "legacy"]
        elif res == "hourly":
            self.data_sources = ["kinergy", "dh"]
        else:
            raise ValueError(f"Unknown value for resolution: {res}")
        self.df = pl.DataFrame()
        self.df_meta = pl.DataFrame()
        self.name = self.res
        self.file_path = PROCESSED_DATA_DIR / f"dataset_{self.name}.csv"

    def create(self):
        dfs = list()
        cols = ["id", "datetime", "diff", "value"]
        logger.info(f"Creating {self.res} dataset")
        for data_source in self.data_sources:
            dfs.append(pl.read_csv(INTERIM_DATA_DIR / f"{data_source}_{self.res}.csv").select(cols).with_columns(
                pl.lit(data_source).alias("source")))
        df = pl.concat(dfs)
        logger.info(f"Number of rows: {df.shape[0]}")
        n_sensors = len(df.group_by(pl.col("id")).agg())
        logger.info(f"Number of sensors: {n_sensors}")
        self.df = df

    def clean(self, plot: bool = False) -> None:
        """
        Clean DataFrame from outliers, negative values, filter connection errors, flat lines, and store missing dates
        :param plot: whether to plot each building with missing dates to FIGURES_DIR
        """
        logger.info(f"Cleaning {self.res} dataset")
        df = self.df
        logger.info(f"Number of rows: {len(df)}")

        logger.info("Removing negative diff values")
        df = remove_neg_diff_vals(df)
        logger.info(f"Number of rows after removing negative values: {len(df)}")

        logger.info("Filter connection error values")
        freq = "D" if self.res == "daily" else "h"
        df = filter_connection_errors_by_id(df, freq)
        logger.info(f"Number of rows after removing connection error values: {len(df)}")

        logger.info("Filtering outliers")
        df = filter_outliers_by_id(df, "diff")
        logger.info(f"Number of rows after filtering outliers: {len(df)}")

        logger.info("Filtering flat lines")
        thresh = THRESHOLD_FLAT_LINES_HOURLY if self.res == "hourly" else THRESHOLD_FLAT_LINES_DAILY
        df = filter_flat_lines_by_id(df, thresh)
        logger.info(f"Number of rows after filtering flat lines: {len(df)}")

        if plot: plot_missing_dates_per_building(df, self.res)
        get_missing_dates(df, frequency=freq)
        logger.info(f"Number of sensors after cleaning data: {len(df.group_by('id').agg())}")
        logger.success(f"Number of rows after cleaning data: {len(df)}")
        self.df = df

    def add_features(self) -> None:
        """
        Add features to dataset. Features include:

        - weather data: weather information from meteostat, stored in data/raw/weather_daily.csv
        - holiday data: whether the day is a holiday in the respective state, stored in data/raw/holidays.csv
        - address data: stored in data/raw/cities.csv
        - building data: stored in the respective meta data files, stored in data/raw/{dataset}_meta.csv, features like
            heated area, number of appartments, type of building (school, gym, museum, multiple appartments, ...)
        - engineered features: "weekend" (whether it is a workday or the weekend), "yearly_consumption" (how much energy
            the building needs a year)
        """
        # load all the feature dataframes
        df_weather = pl.read_csv(FEATURES_DIR / f"weather_{self.res}.csv").with_columns(
            pl.col("time").str.to_datetime().alias("datetime"))
        df_holidays = pl.read_csv(FEATURES_DIR / "holidays.csv").with_columns(pl.col("start").str.to_date(),
                                                                              pl.col("end").str.to_date(strict=False))
        df_cities = pl.read_csv(FEATURES_DIR / f"city_info_{self.res}.csv")

        # META DATA
        df_meta_l = pl.read_csv(META_DIR / "legacy_meta.csv").with_columns(pl.col("plz").str.strip_chars())
        df_meta_dh = pl.read_csv(META_DIR / "dh_meta.csv")
        df_lod = pl.read_csv(META_DIR / "dh_meta_lod.csv").rename(
            {"adresse": "address"})  # dh data with lod building data
        df_meta_dh = df_meta_dh.join(df_lod, on=["address"]).drop(
            ["id_right", "typ_right", "primary_energy_right", "postal_code_right", "city", "postal_code"]
        ).rename({"heated_area": "heated_area_lod"})  # mark heated area feature as generated
        df_meta_k = pl.read_csv(META_DIR / "kinergy_meta.csv", null_values="")
        df_meta = pl.concat(
            [df_meta_l.cast({"plz": pl.Int64}).rename(
                {"qmbehfl": "heated_area", "anzlwhg": "anzahlwhg", "adresse": "address"}).with_columns(
                pl.lit("gas").alias("primary_energy")),
                df_meta_k.rename({"name": "address"}),
                df_meta_dh.rename({"Height (m)": "building_height", "Storeys Above Ground": "storeys_above_ground"})
            ],
            how="diagonal")

        # create holiday dictionary
        holiday_dict = {"BE": [], "HH": [], "MV": [], "BY": [], "SH": []}
        for row in df_holidays.iter_rows():
            if row[1] is not None and row[2] is not None:
                span = pd.date_range(row[1], row[2], freq="D")
                holiday_dict[row[0]].extend(span)
            elif row[1] is not None:
                holiday_dict[row[0]].extend([row[1]])

        attributes = FEATURES_DAILY if self.res == "daily" else FEATURES_HOURLY
        self.df_meta = df_meta

        def is_weekend(date: datetime.datetime):
            if date.weekday() > 4:
                return 1
            else:
                return 0

        def add_holidays(df):
            return df.join(df_cities.select(["plz", "state"]), on="plz", how="left").drop_nulls(["state"]).with_columns(
                pl.struct(["state", "datetime"]).map_elements(
                    lambda x: 1 if x["datetime"] in holiday_dict[x["state"]] else 0,
                    return_dtype=pl.Int64).alias("holiday"))

        def add_features(df):
            enc = LabelEncoder()
            # join data with meta data
            # create new id, needed if data series was split, but belongs to same building
            df = df.with_columns(pl.col("id").str.replace("(-\d+)?-\d+$", "").alias("meta_id")) # TODO
            df = df.join(df_meta.rename({"id": "meta_id"}), on="meta_id", how="left")
            df = (df.join(df_weather.with_columns(pl.col("datetime").dt.cast_time_unit("ns")), on=["datetime", "plz"],
                          how="left")
                  .with_columns(
                # set 0 values in heated area to null
                pl.when(pl.col("heated_area") == 0).then(None).otherwise(pl.col("heated_area")).name.keep(),
                # add weekend column
                pl.col("datetime").map_elements(is_weekend, return_dtype=pl.Int64).alias("weekend"),
                # set all null values of typ to Mehrfamilienhaus
                pl.when(pl.col("typ").is_null()).then(pl.lit("Mehrfamilienhaus")).otherwise(
                    pl.col("typ")).name.keep(),
            ).with_columns(
                # set values in n appartments to null if it is 0 and a if building is multiple appartment building
                pl.when((pl.col("anzahlwhg") == 0).and_(pl.col("typ") == "Mehrfamilienhaus")
                        ).then(None).otherwise(pl.col("anzahlwhg")).name.keep()
            ).with_columns(pl.col("typ").map_batches(enc.fit_transform))  # make typ column categorical
                  )
            # get the daily consumption average for each building
            df_daily_avg = df.select(["meta_id", "datetime", "diff"]).group_by(["meta_id"]).agg(
                pl.col("diff").sum().alias("sum"),
                pl.col("diff").count().alias("count")
            ).with_columns(
                (pl.col("sum") / pl.col("count")).alias("daily_avg")).select("meta_id", "daily_avg")
            df = df.join(df_daily_avg, on="meta_id", how="left")
            df = add_holidays(df)  # TODO: add holidays for more data

            # add cyclic encoded weekdays
            df = df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"),
                                 pl.col("datetime").dt.day().alias("day_of_month"))
            return df

        logger.info(f"Adding {attributes} to dataset, this might take a while")
        self.df = add_features(self.df).select(["id", "datetime"] + attributes)
        logger.success("Added features to dataset")

    def save(self, output_file_path: str):
        logger.info(f"Saving {self.res} dataset of length {len(self.df)} to {output_file_path}")
        self.df.write_csv(output_file_path)

    def load_feat_data(self, interpolate: bool = False):
        if interpolate:
            file_path = f"{PROCESSED_DATA_DIR}/dataset_interpolate_{self.res}_feat.csv"
        else:
            file_path = f"{PROCESSED_DATA_DIR}/dataset_{self.res}_feat.csv"
        self.df = pl.read_csv(file_path).cast(
            {"heated_area": pl.Float64,
             "anzahlwhg": pl.Int64,
             "building_height": pl.Float64,
             "storeys_above_ground": pl.Int64,
             "ground_surface": pl.Float64,
             "tsun": pl.Float64}
        ).with_columns(pl.col("datetime").str.to_datetime())

    def create_and_clean(self, plot: bool = False):
        self.create()
        self.clean(plot=plot)
        self.save(output_file_path=f"{PROCESSED_DATA_DIR}/dataset_{self.name}.csv")

    def create_clean_and_add_feat(self):
        if os.path.exists(self.file_path):
            self.df = pl.read_csv(self.file_path).with_columns(pl.col("datetime").str.to_datetime()).with_columns(
                pl.col("datetime").dt.cast_time_unit("ns"))
        else:
            self.create()
            self.clean()
        self.add_features()
        self.save(output_file_path=f"{PROCESSED_DATA_DIR}/dataset_{self.name}_feat.csv")


class InterpolatedDataset(Dataset):
    def __init__(self, res: str = "daily"):
        super().__init__(res)
        self.name = f"interpolate_{res}"
        self.file_path = PROCESSED_DATA_DIR / f"dataset_{self.name}.csv"

    @overrides
    def clean(self, plot: bool = False):
        super().clean(plot=plot)
        df = self.df
        logger.info("Split series with long series of missing values")
        df = df.with_columns(pl.col("datetime").dt.cast_time_unit("ns"))
        min_gap_size_periods = MIN_GAP_SIZE_DAILY if self.res == "daily" else MIN_GAP_SIZE_HOURLY
        df = split_series_by_id_list(df, min_gap_size_periods, res=self.res, plot=False)  # split series if there are long missing periods
        logger.info("Interpolating values")
        freq = "D" if self.res == "daily" else "h"
        df = interpolate_values_by_id(df, freq)  # interpolate missing dates/hours
        logger.info("Remove negative consumption values")
        df = remove_neg_diff_vals(df)  # interpolation might create new negative diff values, remove them
        df = remove_positive_jumps(df)  # remove too high diff values
        logger.info("Split series with long series of missing values")
        df = split_series_by_id(df, 1, self.res, plot)  # split series if there were holes created by removing neg diff values
        logger.info(f"Number of rows after interpolating: {len(df)}")

        df_info = df.group_by("id").agg(pl.len()).to_pandas()
        df_info.loc["mean"] = [None, df_info["len"].mean()]
        df_info.loc["median"] = [None, df_info["len"].median()]
        df_info.loc["std"] = [None, df_info["len"].std()]
        output_path_info = REPORTS_DIR / f"dataset_{self.name}_{min_gap_size_periods}_series_info.csv"
        df_info.to_csv(output_path_info)
        logger.info(f"Saved series info to {output_path_info}")
        logger.info(f"Number of series after splitting: {len(df_info)-1}")

        assert len(df.filter(pl.col("diff") < 0)) == 0  # no negative diff values should be present
        self.df = df


class TrainingDataset(Dataset):
    """ Dataset for training a model"""

    def __init__(self, config: dict):
        # scaling parameters
        self.df_scaled = None  # store scaled dataframe
        self.scaler_y = None
        self.scaler_X = None
        self.cont_features = None
        self.static_features = None
        self.scale = False  # whether scalers are fit / needed
        self.X_test_scaled = None
        self.y_test_scaled = None

        self.discarded_ids = list()  # ids discarded during training test split because of insufficient length
        self.s_ids = list()  # sensor ids without cut suffix and without discarded ids

        try:
            res = config["res"]
        except KeyError:
            res = "daily"
        super().__init__(res)
        # set the features according to feature code
        try:
            logger.info(f"Features: {config['features']}")
        except KeyError:
            config["features"] = FEATURE_SETS[config["feature_code"]]
        config["features"].sort()  # sort alphabetically
        self.config = config
        self.corrupt_building_ids = [
            ""
        ]

        # train test val split (pandas DataFrames)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        # indexes for each part of the split, correlating to self.df
        self.train_idxs = list()
        self.test_idxs = list()
        self.val_idxs = list()

        self.meta_features = self.config["features"]  # features that need to be not null

        # mapping of ids to test X and y tuples
        self.id_to_test_series = dict()
        self.id_to_test_series_scaled = dict()

        self.name = (f"{self.config['dataset']}_{'interpolate' if self.config['interpolate'] else 'no_interpolate'}"
                     f"_{self.res}_lag_{self.config['lag_in']}_{self.config['lag_out']}")
        self.file_path = f"{PROCESSED_DATA_DIR}/null.csv"  # should not exist to trigger create and clean function

    def get_from_idxs(self, data_split: str, scale: bool = False) -> pl.DataFrame:
        """
        Given the name of the split (either, "train", "test", or "val), return the corresponding data for training/
        testing indexes. Returns all features.
        :param scale: whether to scale data
        :param data_split:
        :return:
        """
        match data_split:
            case "train":
                idxs = self.train_idxs
            case "test":
                idxs = self.test_idxs
            case "val":
                idxs = self.val_idxs
            case _:
                idxs = None
        if idxs is None: raise ValueError("Invalid value for data_split")
        if scale:
            df = self.df_scaled.filter(pl.col("index").is_in(idxs))
        else:
            df = self.df.filter(pl.col("index").is_in(idxs))
        return df

    def get_train_df(self, scale: bool = False) -> pl.DataFrame:
        return self.get_from_idxs("train", scale)

    def get_test_df(self, scale: bool = False) -> pl.DataFrame:
        """
        Get the data of the test split with all columns, not only feature data
        """
        return self.get_from_idxs("test", scale)

    def get_val_df(self, scale: bool = False) -> pl.DataFrame:
        return self.get_from_idxs("val", scale)

    def get_heated_area_by_id(self, id: str) -> float:
        return self.df.filter(pl.col("id") == id)["heated_area"].first()

    @overrides
    def create(self):
        super().load_feat_data(self.config["interpolate"])

    @overrides
    def clean(self, plot: bool = False) -> None:
        pass  # already cleaned

    @overrides
    def add_features(self) -> None:
        self.preprocess()  # updates df and config with training features

    def one_hot_encode(self):
        """
        One hot encode categorical features. Updates config with new feature names.
        """
        df = self.df.to_pandas()
        config = self.config
        enc = OneHotEncoder()
        cat_features = list(set(config["features"]) & set(CATEGORICAL_FEATURES))  # categorical features we want
        if len(cat_features) > 0:
            enc = enc.fit(df[cat_features])
            cat_features_names = enc.get_feature_names_out()
            X_enc = DataFrame(enc.transform(df[cat_features]).toarray(), columns=cat_features_names)
            df = df.drop(columns=cat_features)
            df = pd.concat([df, X_enc], axis=1)
            self.df = pl.DataFrame(df)
            try:
                config["features"] = list(set(config["features"]) - set(cat_features)) + list(cat_features_names)
            except wandb.sdk.lib.config_util.ConfigError:
                wandb.config.update({"features": list(set(config["features"]) - set(cat_features)) + list(cat_features_names)}, allow_val_change=True)
                config = wandb.config
        self.config = config

    def remove_corrupt_buildings(self):
        self.df = self.df.filter(~pl.col("id").is_in(self.corrupt_building_ids))

    def encode_cyclic_features(self):
        """
        Encode cyclic features used for training with sine and cosine functions. Update features in config variable.
        """
        fs = list(set(self.config["features"]) & set(CONTINUOUS_FEATURES_CYCLIC))
        if len(fs) > 0:
            for f in fs:
                self.df = self.df.with_columns(((2 * math.pi * pl.col(f)) / 24).sin().alias(f"{f}_sin"),
                                               ((2 * math.pi * pl.col(f)) / 24).cos().alias(f"{f}_cos"))
                self.df = self.df.drop(f)
            cyclic_encoded_features = (list(set(self.config["features"]) - set(fs)) + [f"{f}_sin" for f in fs] + [f"{f}_cos" for f in fs])
            try:
                self.config["features"] = cyclic_encoded_features
            except:
                wandb.run.config.update({"features": cyclic_encoded_features}, allow_val_change=True)
                self.config = wandb.run.config

    def fit_scalers(self):
        """
        Fits scalers based on the configuration provided. This method initializes and
        fits feature and target scalers for transforming training data based on the
        specified scaling mode. It supports two scaling modes: "all", where a single
        scaler is used globally across the dataset, and "individual", where separate
        scalers are used for different subsets of the data. The function also handles
        transformations for the training data post-fitting.

        Raises an error if an unknown scaling mode is specified. If called when scalers
        are already fitted, the method exits without re-fitting.

        :raises ValueError: If an unknown scaling mode is specified.
        """
        if self.scaler_X is not None or self.scaler_y is not None:
            return  # already fitted

        if self.config["scaler"] == "none":
            return

        config = self.config
        scale_mode = config.get("scale_mode", "individual")

        self._set_features()

        if scale_mode == "all":
            self.scaler_X, self.scaler_y = self._create_scaler_pair()
            if self.scaler_X and self.cont_features:
                # continuous features are used in training
                self.scaler_X.fit(self.X_train[self.cont_features])
            self.scaler_y.fit(self.y_train["diff"].to_numpy().reshape(-1, 1))
            self._transform_all_data()

        elif scale_mode == "individual":
            self.s_ids = self._get_scaler_ids()
            self.scaler_X, self.scaler_y = self._create_scaler_pair()
            if self.scaler_X and self.cont_features:
                # continuous features are used in training
                self.scaler_X.fit(self.X_train[self.cont_features])
            train_df = self.get_train_df()
            for s_id in self.s_ids:
                df_filter = train_df.filter(pl.col("id").str.starts_with(s_id))
                if len(df_filter) > 0:
                    self.scaler_y[s_id].fit(df_filter["diff"].to_numpy().reshape(-1, 1))
                else:
                    # the id is not used
                    del self.scaler_y[s_id]
                    self.discarded_ids.append(s_id)
            self._transform_individual_data()
        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        self.scale = True

    def _create_scaler_pair(self):
        """
        Create scaler instances based on the scaler type.
        If n_ids is provided, create a list of scalers for each id.

        :return: scaler_X and scaler_y
        """
        scaler_type = self.config.get("scaler", "none")
        if scaler_type == "none":
            return None, None

        scaler_cls = MinMaxScaler if scaler_type == "minmax" else StandardScaler

        if self.s_ids is None:
            return scaler_cls(), scaler_cls()
        else:
            return scaler_cls(), {s_id: scaler_cls() for s_id in self.s_ids}

    def _set_features(self):
        """Set continuous features used for training and static features."""
        cont_features = list(set(self.config["features"]) & set(CONTINUOUS_FEATURES))
        self.cont_features = cont_features
        self.static_features = list(set(self.config["features"]) - set(["diff"] + cont_features))

    def _get_scaler_ids(self):
        """
        Retrieve unique original IDs from the dataframe, excluding discarded IDs.

        :returns: List of scaler IDs
        """
        s_ids = self.df.with_columns(
            pl.col("id").str.replace_all("(-\d+)?-\d+$", "").alias("orig_id")
        )["orig_id"].unique().to_list()
        return [x for x in s_ids if x not in self.discarded_ids]

    def _transform_all_data(self):
        """
        Transform the entire dataset using fitted scalers for all data.
        """
        df = self.df.to_pandas()
        if self.scaler_X and self.cont_features:
            df[self.cont_features] = self.scaler_X.transform(df[self.cont_features])
        if self.scaler_y:
            df["diff"] = self.scaler_y.transform(df["diff"].to_numpy().reshape(len(df), 1))
        assert len(df) == len(self.df)
        assert (df["index"].sort_values() == self.df[
            "index"].to_pandas().sort_values()).all()  # both dataframes should have same datapoints
        self.df_scaled = pl.DataFrame(df)

    def _transform_individual_data(self):
        """
        Transform the dataset by applying individual scalers for each ID.

        :raises sklearn.exceptions.NotFittedError: If any scaler for target `diff` has not been
            fitted prior to the transformation.
        :raises AssertionError: If the length of the training indices `self.train_idxs` does not
            match the filtered and scaled data.
        """
        df = self.df.to_pandas()
        scaled_dfs = []

        for s_id, scaler in self.scaler_y.items():
            df_filter = df[df["id"].str.startswith(s_id)].copy()
            if len(df_filter) > 0:
                if self.scaler_X and self.cont_features:
                    df_filter[self.cont_features] = self.scaler_X.transform(df_filter[self.cont_features])
                df_filter["diff"] = scaler.transform(df_filter["diff"].to_numpy().reshape(-1, 1))
                scaled_dfs.append(pl.DataFrame(df_filter))

        self.df_scaled = pl.concat(scaled_dfs, how="vertical")

        # make sure that all needed datapoints are present in scaled dataframe
        assert len(set(self.df_scaled["index"].to_list()).intersection(set(self.train_idxs))) == len(self.train_idxs)
        assert len(set(self.df_scaled["index"].to_list()).intersection(set(self.test_idxs))) == len(self.test_idxs)
        assert len(set(self.df_scaled["index"].to_list()).intersection(set(self.val_idxs))) == len(self.val_idxs)

    def handle_missing_features(self):
        # check if needed features are all in the dataset
        if not set(self.meta_features).issubset(set(self.df.columns)):
            raise ValueError(f"Features {self.config['features']} not in dataset")
        self.df = self.df.drop_nulls(subset=self.meta_features)  # remove null values for used features

    def drop_lag_feature_nulls(self):
        """
        Drop rows where needed lag features are null. Whether a lag feature is needed is determined by
        the dataset configuration lag_in, lag_out. If these are not given, use n_in, n_out.
        """
        try:
            n_in, n_out = self.config["lag_in"], self.config["lag_out"]
            assert n_in >= self.config["n_in"] and n_out >= self.config["n_out"]
        except KeyError:
            n_in, n_out = self.config["n_in"], self.config["n_out"]
        self.df = self.df.drop_nulls([f"diff(t-{i})" for i in range(1, n_in + 1)])
        self.df = self.df.drop_nulls([f"diff(t+{i})" for i in range(1, n_out)])

    def create_lag_features(self):
        """
        Add lag features to the data.
        :return: None
        """
        for i in range(1, self.config["lag_in"] + 1):  # create diff for past n days
            self.df = self.df.with_columns(pl.col("diff").shift(i).over("id").alias(f"diff(t-{i})"))
        for i in range(1, self.config["lag_out"]):  # t=0 already in data
            self.df = self.df.with_columns(pl.col("diff").shift(-i).over("id").alias(f"diff(t+{i})"))

    def preprocess(self) -> tuple[pl.DataFrame, dict]:
        """
        Preprocessing of data for model training.
        :return: preprocessed DataFrame and updated config dictionary
        """
        self.create_lag_features()
        self.drop_lag_feature_nulls()
        # select energy type
        if self.config["energy"] != "all":
            self.df = self.df.filter(pl.col("primary_energy") == self.config["energy"])
        self.handle_missing_features()
        self.one_hot_encode()  # one hot encode after removing features
        self.remove_corrupt_buildings()
        self.encode_cyclic_features()
        logger.info(f"Training Features: {self.config['features']}")
        # self.save(PROCESSED_DATA_DIR / f"{self.config['dataset']}_{self.config['lag_in']}_{self.config['lag_out']}.csv")
        return self.df, self.config

    def compute_clusters(self) -> dict:
        """
        Compute cluster mappings with meta data
        :return: dictionary mapping each cluster ID to a list of test ID indexes
        """
        n_clusters = N_CLUSTER
        df = hierarchical_clustering_on_meta_data(self.df, n_clusters)
        cluster_id_map = dict()  # map each id to a cluster
        for row in df.iter_rows():
            cluster_id_map[row[0]] = row[-1]
        df = self.df.sort(by=["id", "datetime"]).select(pl.col("id")).with_row_index()
        # get only test data
        df_test = df.filter(pl.col("index").is_in(self.test_idxs)).with_row_index("test_idx")
        # map each id to cluster label
        df_test = df_test.with_columns(pl.col("id").replace(cluster_id_map).alias("label"))
        # create cluster mapping each cluster label to list of test indexes
        cluster_map = dict()
        for label in df_test["label"].unique():
            cluster_map[label] = df_test.filter(pl.col("label") == label)["test_idx"].to_list()
        return cluster_map

    def _rescale_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        b_id = df["id"].first()
        index_column = df["index"]
        df = df.drop(["id", "index"])
        scaler_key = re.sub(r'(-\d+)*$', '', b_id)  # get base id without suffixes
        rescaled_df = self.scaler_y[scaler_key].inverse_transform(df.to_numpy().reshape(-1, self.config["n_out"]))
        return pl.DataFrame(rescaled_df).with_columns(pl.lit(b_id).alias("id")).insert_column(0, index_column)

    def rescale_predictions(self, y_hat_scaled: np.ndarray, id_column: pl.Series) -> np.ndarray:
        if self.config["scale_mode"] == "all":
            return self.scaler_y.inverse_transform(y_hat_scaled.reshape(len(y_hat_scaled), self.config["n_out"]))
        else:  # TODO
            df = pl.DataFrame(y_hat_scaled).insert_column(0, id_column).with_row_index()
            rescaled_df = df.group_by("id").map_groups(lambda group: self._rescale_predictions(group))
            y_hat = rescaled_df.sort(by=["index"]).drop(["id", "index"]).to_numpy()
            return y_hat


class TrainDataset90(TrainingDataset):
    def __init__(self, config: dict):
        super().__init__(config)

    def remove_corrupt_buildings(self):
        frequ = "D" if self.res == "daily" else "h"
        md_df = get_missing_dates(self.df, frequ)
        logger.info("Filtering for at most 10% missing data")
        allowed_ids = md_df.filter(pl.col("per") > 90)["id"].to_list()
        self.df = self.df.filter(pl.col("id").is_in(allowed_ids))
        logger.info(f"Data length after filtering: {len(self.df)}")


class TrainDatasetMeta(TrainingDataset):
    """ Dataset for training the model with data that has meta data available """

    def __init__(self, config: dict):
        super().__init__(config)
        self.meta_features = FEATURE_SET_8  # diff + weather + time + daily_avg


class TrainDatasetBuilding(TrainingDataset):
    """ Dataset for training the model with data that has meta and building info available """

    def __init__(self, config: dict):
        super().__init__(config)
        if self.res == "daily":
            self.meta_features = FEATURE_SET_10  # diff + weather + time + building (no apartment)
        elif self.res == "hourly":
            self.meta_features = FEATURE_SET_15  # same as 10 but with hourly weather features





if __name__ == '__main__':
    # DATA LOADING
    logger.info("Start data loading")

    # daily data
    # LegacyDataLoader(RAW_DATA_DIR / "legacy_data" / "legacy_systen_counter_daily_values.csv").write_data_and_meta()
    # KinergyDataLoader(RAW_DATA_DIR / "kinergy").write_data_and_meta()
    # DHDataLoader(RAW_DATA_DIR / "district_heating_data").write_data_and_meta()
    #
    # # hourly data
    # KinergyDataLoader(RAW_DATA_DIR / "kinergy", res="hourly").write_data_and_meta()
    # DHDataLoader(RAW_DATA_DIR / "district_heating_data", res="hourly").write_data_and_meta()

    # meta data

    logger.info("Finish data loading")

    ds = InterpolatedDataset(res="hourly")
    # ds.create_and_clean(plot=True)
    ds.create_clean_and_add_feat()
    # #
    # ds = Dataset(res="hourly")
    # ds.create_and_clean(plot=True)
    # ds.create_clean_and_add_feat()

    # ds.load_feat_data()
    # df_train, df_test = ds.get_train_and_test(0.8)
    # ds.create_and_clean()

    # ds_hourly = Dataset(res="hourly")
    # ds_hourly.create_and_clean()

    # freeze dataset for certain config
    freeze_config = {"project": "ma-wahl-forecast",
                     "energy": "all",
                     "res": "daily",
                     "interpolate": 1,
                     "dataset": "building",  # building, meta, missing_data_90
                     "model": "RNN1",
                     "lag_in": 7,
                     "lag_out": 7,
                     "n_in": 7,
                     "n_out": 7,
                     "n_future": 7,
                     "scaler": "none",
                     "scale_mode": "all",  # all, individual
                     "feature_code": 13,
                     "train_test_split_method": "time"}  # ReLU, Linear

    # ds = TrainDatasetBuilding(freeze_config)
    # ds.create_clean_and_add_feat()
