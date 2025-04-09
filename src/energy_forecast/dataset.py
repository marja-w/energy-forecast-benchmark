import datetime
import math
import os
from argparse import ArgumentError
from itertools import product

import darts
import numpy as np
import pandas as pd
import polars as pl
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from loguru import logger
from overrides import overrides
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

from src.energy_forecast.config import DATA_DIR, PROCESSED_DATA_DIR, CATEGORICAL_FEATURES, FEATURES, \
    FEATURES_DIR, META_DIR, INTERIM_DATA_DIR, N_CLUSTER, REPORTS_DIR, CONTINUOUS_FEATURES_CYCLIC, CONTINUOUS_FEATURES
from src.energy_forecast.plots import plot_missing_dates_per_building, plot_clusters
from src.energy_forecast.utils.cluster import hierarchical_clustering_on_meta_data
from src.energy_forecast.utils.data_processing import remove_neg_diff_vals, filter_connection_errors_by_id, \
    filter_outliers_by_id, filter_flat_lines_by_id, split_series_by_id_list, interpolate_values_by_id, \
    remove_positive_jumps, split_series_by_id, remove_null_series_by_id
from src.energy_forecast.utils.time_series import sensors_to_supervised, series_to_supervised
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
        thresh = 12 if self.res == "hourly" else 7  # allowed length of flat lines
        df = filter_flat_lines_by_id(df, thresh)
        logger.info(f"Number of rows after filtering flat lines: {len(df)}")

        if plot: plot_missing_dates_per_building(df)
        get_missing_dates(df)
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
        df_weather = pl.read_csv(FEATURES_DIR / "weather_daily.csv").with_columns(
            pl.col("time").str.to_datetime().alias("datetime"))
        df_holidays = pl.read_csv(FEATURES_DIR / "holidays.csv").with_columns(pl.col("start").str.to_date(),
                                                                              pl.col("end").str.to_date(strict=False))
        df_cities = pl.read_csv(FEATURES_DIR / "city_info.csv")

        # META DATA
        df_meta_l = pl.read_csv(META_DIR / "legacy_meta.csv").with_columns(pl.col("plz").str.strip_chars())
        df_meta_dh = pl.read_csv(META_DIR / "dh_meta.csv").rename({"eco_u_id": "id"})
        df_lod = pl.read_csv(META_DIR / "dh_meta_lod.csv").rename(
            {"adresse": "address"})  # dh data with lod building data
        df_meta_dh = df_meta_dh.join(df_lod, on=["address"]).drop(
            ["id_right", "postal_code_right", "city", "postal_code"])
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

        attributes = FEATURES
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
            df = df.with_columns(pl.col("id").str.replace("(-\d)?-\d$", "").alias("meta_id"))
            df = df.join(df_meta.rename({"id": "meta_id"}), on="meta_id", how="left")
            logger.info(f"Data length after joining with meta data: {len(df)}")
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
            logger.info(f"Data length after joining with weather data: {len(df)}")
            # get the daily consumption average for each building
            df_daily_avg = df.select(["meta_id", "datetime", "diff"]).group_by(["meta_id"]).agg(
                pl.col("diff").sum().alias("sum"),
                pl.col("diff").count().alias("count")
            ).with_columns(
                (pl.col("sum") / pl.col("count")).alias("daily_avg")).select("meta_id", "daily_avg")
            df = df.join(df_daily_avg, on="meta_id", how="left")
            logger.info(f"Data length after joining with daily average data: {len(df)}")
            df = add_holidays(df)  # TODO: add holidays for more data

            # add cyclic encoded weekdays
            df = df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"),
                                 pl.col("datetime").dt.day().alias("day_of_month"))
            return df

        logger.info(f"Adding {attributes} to dataset, this might take a while")
        # create diff of past day feature
        self.df = self.df.with_columns(pl.col("diff").shift(1).over("id").alias("diff_t-1")).drop_nulls(
            subset=["diff_t-1"])
        self.df = add_features(self.df).select(["id", "datetime"] + attributes)
        logger.success("Added features to dataset")

    def get_train_and_test(self, train_per: float):
        gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_per, random_state=42)
        df = self.df.with_row_index()
        for train_idx, test_idx in gss.split(df, groups=df["id"]):
            train_data = df.filter(pl.col("index").is_in(train_idx))
            test_data = df.filter(pl.col("index").is_in(test_idx))
        return train_data, test_data

    def save(self, output_file_path: str):
        logger.info(f"Saving {self.res} dataset to {output_file_path}")
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
        super().clean(plot=False)
        df = self.df
        logger.info("Split series with long series of missing values")
        df = df.with_columns(pl.col("datetime").dt.cast_time_unit("ns"))
        df = split_series_by_id_list(df, 7, plot=False)  # split series if there are long missing periods
        logger.info("Interpolating values")
        df = interpolate_values_by_id(df)  # interpolate missing dates/hours
        logger.info("Remove negative consumption values")
        df = remove_neg_diff_vals(df)  # interpolation might create new negative diff values, remove them
        df = remove_positive_jumps(df)  # remove too high diff values
        logger.info("Split series with long series of missing values")
        df = split_series_by_id(df, 1, plot)  # split series if there were holes created by removing neg diff values
        logger.info(f"Number of rows after interpolating: {len(df)}")
        assert len(df.filter(pl.col("diff") < 0)) == 0  # no negative diff values should be present
        self.df = df


class TrainingDataset(Dataset):
    """ Dataset for training a model"""

    def __init__(self, config: dict):
        # scaling parameters
        self.scaler_y = None
        self.scaler_X = None
        self.cont_features = None
        self.static_features = None
        self.scale = False  # whether scalers are fit / needed
        self.X_test_scaled = None
        self.y_test_scaled = None
        try:
            res = config["res"]
        except KeyError:
            res = "daily"
        super().__init__(res)
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
        df = self.df.sort(by=["id", "datetime"]).with_row_index()
        df = df.filter(pl.col("index").is_in(idxs))
        if scale:
            df = df.to_pandas()
            if self.scaler_X is not None:  # if only diff is feature, we dont scale other features
                df[self.cont_features] = self.scaler_X.transform(df[self.cont_features])
            df["diff"] = self.scaler_y.transform(df["diff"].to_numpy().reshape(len(df), 1))
            df = pl.DataFrame(df)
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
            config["features"] = list(set(config["features"]) - set(cat_features)) + list(cat_features_names)
        self.config = config

    def add_multiple_forecast(self) -> None:
        """
        Add heat consumption of next n days/hours to data, if "n_out" greater than 1.
        """
        n = self.config["n_out"]
        if n > 1:
            df = self.df
            for i in range(1, n):
                df = df.with_columns(pl.col("diff").shift(-i).alias(f"diff(t+{i})"))
            df = df.drop_nulls(["diff"] + [f"diff(t+{i})" for i in range(1, n)])
            self.df = df

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
            self.config["features"] = (list(set(self.config["features"]) - set(fs))
                                       + [f"{f}_sin" for f in fs] + [f"{f}_cos" for f in fs])

    def fit_scalers(self):
        """
        Fit the scalers according to the scaler stored in config on the train split.
        Set continous features used for this training.
        """
        config = self.config

        # set scaler if not set yet
        if self.scaler_X is None and self.scaler_y is None:
            if config["scaler"] == "minmax":
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
            elif config["scaler"] == "standard":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
            elif config["scaler"] == "none":
                self.scaler_X = None
                self.scaler_y = None
                pass
            else:
                raise NotImplementedError(f"Scaler {config['scaler']} not implemented")

            # fit scaler
            cont_features = list(set(config["features"]) & set(CONTINUOUS_FEATURES))
            if len(cont_features) > 0:  # only if there are other continuous features other than diff
                self.scaler_X = scaler_X.fit(self.X_train[cont_features])
                self.cont_features = cont_features
                self.static_features = list(set(self.config["features"]) - set(["diff"] + self.cont_features))
            # target variable
            self.scaler_y = scaler_y.fit(self.y_train["diff"].to_numpy().reshape(len(self.y_train), 1))
            self.scale = True
        else:  # already fitted
            pass

    def handle_missing_features(self):
        self.df = self.df.drop_nulls(subset=self.config["features"])  # remove null values for used features

    def preprocess(self) -> tuple[pl.DataFrame, dict]:
        """
        Preprocessing of data for model training.
        :return: preprocessed DataFrame and updated config dictionary
        """
        self.add_multiple_forecast()
        # select energy type
        if self.config["energy"] != "all":
            self.df = self.df.filter(pl.col("primary_energy") == self.config["energy"])
        self.one_hot_encode()
        self.handle_missing_features()
        logger.info(f"Training Features: {self.config['features']}")
        self.remove_corrupt_buildings()
        self.encode_cyclic_features()
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
        list[darts.TimeSeries], list[darts.TimeSeries]]:
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

        covariate_list = darts.TimeSeries.from_group_dataframe(df.to_pandas(),
                                                               group_cols="id",
                                                               time_col="datetime",
                                                               value_cols=self.cont_features,
                                                               freq="D",
                                                               fill_missing_dates=True
                                                               )

        # scaling
        if scale:
            scaler_y = Scaler(self.scaler_y, global_fit=True)
            scaler_X = Scaler(self.scaler_X, global_fit=True)

            if data_split == "train":
                target_series = scaler_y.fit_transform(target_series)
                covariate_list = scaler_X.fit_transform(covariate_list)
                self.scaler_y = scaler_y
                self.scaler_X = scaler_X
            else:
                target_series = self.scaler_y.transform(target_series)
                covariate_list = self.scaler_X.transform(covariate_list)

        # fill missing values using darts  # TODO maybe throw away rather than fill?
        transformer = MissingValuesFiller()
        target_series = transformer.transform(target_series)
        covariate_list = transformer.transform(covariate_list)

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

    logger.info("Finish data loading")

    ds = InterpolatedDataset()
    ds.create_and_clean(plot=False)
    ds.create_clean_and_add_feat()

    ds = Dataset()
    ds.create_and_clean()
    # ds.create_clean_and_add_feat()

    # ds.load_feat_data()
    # df_train, df_test = ds.get_train_and_test(0.8)
    # ds.create_and_clean()

    # ds_hourly = Dataset(res="hourly")
    # ds_hourly.create_and_clean()
