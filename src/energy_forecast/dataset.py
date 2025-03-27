import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger
from overrides import overrides
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas import DataFrame

from src.energy_forecast.config import RAW_DATA_DIR, DATA_DIR, PROCESSED_DATA_DIR, CATEGORICAL_FEATURES, FEATURES
from src.energy_forecast.data_processing.data_source import LegacyDataLoader, KinergyDataLoader, DHDataLoader
from src.energy_forecast.plots import plot_missing_dates_per_building, plot_clusters
from src.energy_forecast.util import get_missing_dates, find_time_spans


def remove_neg_diff_vals(df):
    """
    Remove faulty gas meter data points that caused negative diff values
    :param df:
    :return:
    """

    # die diffs stimmen jetzt nicht mehr, wenn reihen entfernt werden. Problem?
    # die diffs sollten weiterhin stimmen, da die differenz vom "falschen" Gaszählerstand immer noch die richtige
    # differenz ist

    return df.filter(
        pl.col("diff") >= 0  # remove all rows with negative usage
    )


def filter_outliers_iqr(df, column):
    """
    Filter outliers in the specified column of the DataFrame using the 1.5 IQR method.

    :param df: polars DataFrame
    :param column: column name to filter outliers
    :return: DataFrame with outliers removed
    """
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25

    upper_bound = q75 + 1.5 * iqr

    filtered_df = df.filter(pl.col(column) <= upper_bound)
    filtered_count = len(df) - len(filtered_df)

    logger.info(f"Filtered {filtered_count} rows for column {column} for ID {df['id'][0]}")

    return filtered_df


def filter_outliers_by_id(df, filter_column):
    """
    Apply the filter_outliers_iqr function to each subset of the DataFrame grouped by the id_column.

    :param df: polars DataFrame
    :param filter_column: column name to filter outliers
    :return: DataFrame with outliers removed for each group
    """

    filtered_df = df.group_by("id").map_groups(lambda group: filter_outliers_iqr(group, filter_column))
    return filtered_df


def filter_flat_lines(df: pl.DataFrame, thresh: int) -> pl.DataFrame:
    df = df.sort(pl.col("datetime"))  # make sure it is in right order
    starting_n = len(df)
    df = df.with_row_index()
    # get a list of number of cumulative zeros and their ending index
    cum_zeros = df.with_columns(
        pl.col("diff").cum_count().over(
            pl.when(pl.col("diff") != 0).then(1).cum_sum().forward_fill()) - 1
    ).filter(pl.col("diff") > thresh  # filter everything longer than threshold
             ).select(["index", "diff"])

    for row in cum_zeros.iter_rows():
        start_idx = row[0] - (row[1] - 1)
        end_idx = row[0] + 1
        df = df.filter(~(pl.col("index").is_in(range(start_idx, end_idx))))

    df = df.drop(["index"])
    logger.info(f"Filtered {starting_n - len(df)} rows for ID {df['id'][0]}")
    return df


def filter_connection_errors(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    start_n = len(df)
    df = df.with_row_index()
    df = df.with_columns(pl.col("datetime").str.to_datetime())
    m_dates = get_missing_dates(df, frequency=freq)["missing_dates"].to_list()[0]
    delta = timedelta(days=1) if freq == "D" else timedelta(hours=1)
    spans = find_time_spans(m_dates, delta)

    if freq == "h":
        spans = spans.filter(pl.col("n") > 2)  # only remove if span big enough

    for row in spans.iter_rows():
        start = row[0]
        end = row[1]
        n = row[2]
        error_val_idx = df.filter(pl.col("datetime") == end + delta)["index"].item()
        df = df.filter(~(pl.col("index") == error_val_idx))  # remove row with erroneous value TODO: interpolate?

    df = df.drop(["index"])
    logger.info(f"Filtered {start_n - len(df)} rows for ID {df['id'][0]}")
    return df


def filter_flat_lines_by_id(df: pl.DataFrame, thresh: int) -> pl.DataFrame:
    filtered_df = df.group_by("id").map_groups(lambda group: filter_flat_lines(group, thresh))
    return filtered_df


def filter_connection_errors_by_id(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    filtered_df = df.group_by("id").map_groups(lambda group: filter_connection_errors(group, freq))
    return filtered_df


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
        self.name = self.res

    def create(self):
        dfs = list()
        cols = ["id", "datetime", "diff"]
        logger.info(f"Creating {self.res} dataset")
        for data_source in self.data_sources:
            dfs.append(pl.read_csv(RAW_DATA_DIR / f"{data_source}_{self.res}.csv").select(cols).with_columns(
                pl.lit(data_source).alias("source")))
        df = pl.concat(dfs)
        logger.info(f"Number of rows: {df.shape[0]}")
        n_sensors = len(df.group_by(pl.col("id")).agg())
        logger.info(f"Number of sensors: {n_sensors}")
        self.df = df

    def clean(self):
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
        thresh = 14 * 24 if self.res == "hourly" else 14  # allowed length of flat lines
        df = filter_flat_lines_by_id(df, thresh)
        logger.info(f"Number of rows after filtering flat lines: {len(df)}")

        plot_missing_dates_per_building(df)
        df_md = get_missing_dates(df)
        df_md.select(["id", "len", "n", "per", "start_date", "end_date"]).sort(pl.col("per"), descending=True).write_csv(RAW_DATA_DIR / "missing_dates.csv")
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
        df_weather = pl.read_csv(RAW_DATA_DIR / f"weather_daily.csv").with_columns(
            pl.col("time").str.to_datetime().alias("datetime"))
        df_holidays = pl.read_csv(RAW_DATA_DIR / "holidays.csv").with_columns(pl.col("start").str.to_date(),
                                                                              pl.col("end").str.to_date(strict=False))
        df_cities = pl.read_csv(RAW_DATA_DIR / "cities.csv")

        # META DATA
        df_meta_l = pl.read_csv(RAW_DATA_DIR / "legacy_meta.csv").with_columns(pl.col("plz").str.strip_chars())
        df_meta_dh = pl.read_csv(RAW_DATA_DIR / "dh_meta.csv").rename({"eco_u_id": "id"})
        df_lod = pl.read_csv(RAW_DATA_DIR / "dh_meta_lod.csv").rename(
            {"adresse": "address"})  # dh data with lod building data
        df_meta_dh = df_meta_dh.join(df_lod, on=["address"]).drop(
            ["id_right", "postal_code_right", "city", "postal_code"])
        df_meta_k = pl.read_csv(RAW_DATA_DIR / "kinergy_meta.csv", null_values="")
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

        def add_meta(df):
            enc = LabelEncoder()
            df = (df.join(df_meta, on="id", how="left")
                  .join(df_weather, on=["datetime", "plz"], how="left")
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
                pl.when((pl.col("anzahlwhg") == 0).and_(pl.col("typ") == "Mehramilienhaus")
                        ).then(None).otherwise(pl.col("anzahlwhg")).name.keep()
            ).with_columns(pl.col("typ").map_batches(enc.fit_transform))  # make typ column categorical
                  )
            # get the daily consumption average for each building
            df_daily_avg = df.select(["id", "datetime", "diff"]).group_by(["id"]).agg(
                pl.col("diff").sum().alias("sum"),
                pl.col("diff").count().alias("count")
            ).with_columns(
                (pl.col("sum") / pl.col("count")).alias("daily_avg")).select("id", "daily_avg")
            df = df.join(df_daily_avg, on="id", how="left")
            return add_holidays(df)

        logger.info(f"Adding {attributes} to dataset, this might take a while")
        # create diff of past day feature
        self.df = self.df.with_columns(pl.col("diff").shift(1).over("id").alias("diff_t-1")).drop_nulls(
            subset=["diff_t-1"])
        self.df = add_meta(self.df).select(["id", "datetime"] + attributes)
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

    def load_feat_data(self):
        self.df = pl.read_csv(f"{PROCESSED_DATA_DIR}/dataset_{self.res}_feat.csv").cast(
            {"heated_area": pl.Float64,
             "anzahlwhg": pl.Int64,
             "max_vorlauf_temp": pl.Float64,
             "min_vorlauf_temp": pl.Float64,
             "building_height": pl.Float64,
             "storeys_above_ground": pl.Int64,
             "ground_surface": pl.Float64,
             "tsun": pl.Float64}
        ).with_columns(pl.col("datetime").str.to_datetime())

    def create_and_clean(self):
        self.create()
        self.clean()
        self.save(output_file_path=f"{PROCESSED_DATA_DIR}/dataset_{self.name}.csv")

    def create_clean_and_add_feat(self):
        self.create()
        self.clean()
        self.add_features()
        self.save(output_file_path=f"{PROCESSED_DATA_DIR}/dataset_{self.name}_feat.csv")


def interpolate_values(df: pl.DataFrame) -> pl.DataFrame:
    freq = "D"  # TODO: for hours
    # create dataframe from start to end date
    start_date = df["datetime"].min()
    end_date = df["datetime"].max()
    df_date_list = pl.DataFrame(pl.from_pandas(pd.date_range(start_date, end_date, freq=freq)), schema={"datetime": pl.Datetime})
    df = df.with_columns(pl.col("datetime").dt.cast_time_unit("ns").alias("datetime"))
    df_dates = df_date_list.join(df, on="datetime", how="left")
    pass  # TODO


def interpolate_values_by_id(df: pl.DataFrame):
    interpolated_df = df.group_by("id").map_groups(lambda group: interpolate_values(group))
    return interpolated_df


class InterpolatedDataset(Dataset):
    def __init__(self, res: str = "daily"):
        super().__init__(res)
        self.name = f"interp_{res}"

    def clean(self):
        super().clean()
        df = self.df
        logger.info("Interpolating values")
        df = interpolate_values_by_id(df)
        logger.info(f"Number of rows after filtering outliers: {len(df)}")
        self.df = df

class TrainingDataset(Dataset):
    """ Dataset for training a model"""

    def __init__(self, config: dict):
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

    def one_hot_encode(self):
        """
        One hot encode categorical features. Returns updated config with new feature names.
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
                df = df.with_columns(pl.col("diff").shift(-i).alias(f"diff_t+{i}"))
            df = df.drop_nulls(["diff"] + [f"diff_t+{i}" for i in range(1, n)])
            self.df = df

    def remove_corrupt_buildings(self):
        self.df = self.df.filter(~pl.col("id").is_in(self.corrupt_building_ids))

    def preprocess(self) -> tuple[pl.DataFrame, dict]:
        self.one_hot_encode()
        self.add_multiple_forecast()
        # select energy type
        if self.config["energy"] != "all":
            self.df = self.df.filter(pl.col("primary_energy") == self.config["energy"])
        self.df = self.df.drop_nulls(subset=self.config["features"])  # remove null values for used features
        logger.info(f"Training Features: {self.config['features']}")
        self.remove_corrupt_buildings()
        return self.df, self.config

    def compute_clusters(self) -> dict:
        """
        Compute cluster mappings with meta data
        :return:
        """
        n_clusters = 3
        logger.info(f"Computing {n_clusters} clusters")
        df = self.df.group_by("id").agg(pl.col("diff"),
                      pl.col("daily_avg").mode().first().alias("avg"),
                      pl.col("diff").std().alias("std"),
                      pl.col("diff").median().alias("median"),
                      pl.col("diff").min().alias("min"),
                      pl.col("diff").max().alias("max"),
                      pl.col("diff").head(30).alias("month"),
                      pl.len()
                      ).sort("len")
        df.drop(["diff", "month"]).write_csv(DATA_DIR / "processed" / "buildings_consumption_info.csv")
        data = df.drop(["id", "diff", "month", "len"]).to_numpy()

        # Create the AgglomerativeClustering model
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

        # Fit the model and predict cluster labels
        labels = model.fit_predict(data)
        plot_clusters(df, labels)

        # add label column to dataframe
        df = df.with_columns(pl.Series(labels).alias("label"))
        for c_id in range(n_clusters):
            logger.info(f"Computed Cluster {c_id} with n={len(df.filter(pl.col('label')==c_id))}")
        cluster_id_map = dict()  # map each id to a cluster
        for row in df.iter_rows():
            cluster_id_map[row[0]] = row[-1]
        df = self.df.select(pl.col("id")).with_row_index()
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
    def add_multiple_forecast(self) -> None:
        """
        Transform dataset into time series data for training.
        :return:
        """
        pass


if __name__ == '__main__':
    # DATA LOADING
    logger.info("Start data loading")

    # daily data
    # LegacyDataLoader(DATA_DIR / "legacy_data" / "legacy_systen_counter_daily_values.csv").write_data_and_meta()
    # KinergyDataLoader(DATA_DIR / "kinergy").write_data_and_meta()
    # DHDataLoader(DATA_DIR / "district_heating_data").write_data_and_meta()
    #
    # # hourly data
    # KinergyDataLoader(DATA_DIR / "kinergy", res="hourly").write_data_and_meta()
    # DHDataLoader(DATA_DIR / "district_heating_data", res="hourly").write_data_and_meta()

    logger.info("Finish data loading")

    ds = Dataset()
    ds.create_and_clean()
    # ds.create_clean_and_add_feat()

    # ds.load_feat_data()
    # df_train, df_test = ds.get_train_and_test(0.8)
    # ds.create_and_clean()

    # ds_hourly = Dataset(res="hourly")
    # ds_hourly.create_and_clean()
