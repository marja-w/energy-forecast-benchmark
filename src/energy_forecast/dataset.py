from datetime import timedelta

import polars as pl
from loguru import logger

from config import RAW_DATA_DIR, DATA_DIR, PROCESSED_DATA_DIR
from src.energy_forecast.data_processing.data_source import LegacyDataLoader, KinergyDataLoader, DHDataLoader
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


# TODO: add features
class Dataset(object):
    def __init__(self, res: str = "daily"):
        self.res = res
        if res == "daily":
            self.data_sources = ["kinergy", "dh", "legacy"]
        elif res == "hourly":
            self.data_sources = ["kinergy", "dh"]
        else:
            raise ValueError(f"Unknown value for resolution: {res}")
        self.df = pl.DataFrame()

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
        thresh = 14*24 if self.res == "hourly" else 14  # allowed length of flat lines
        df = filter_flat_lines_by_id(df, thresh)
        logger.info(f"Number of rows after filtering flat lines: {len(df)}")

        logger.success(f"Number of rows after cleaning data: {len(df)}")
        self.df = df

    def save(self):
        output_file_path = f"{PROCESSED_DATA_DIR}/dataset_{self.res}.csv"
        logger.info(f"Saving {self.res} dataset to {output_file_path}")
        self.df.write_csv(output_file_path)

    def create_and_clean(self):
        self.create()
        self.clean()
        self.save()


if __name__ == '__main__':
    # DATA LOADING
    logger.info("Start data loading")

    # daily data
    LegacyDataLoader(DATA_DIR / "legacy_data" / "legacy_systen_counter_daily_values.csv").write_data_and_meta()
    KinergyDataLoader(DATA_DIR / "kinergy").write_data_and_meta()
    DHDataLoader(DATA_DIR / "district_heating_data").write_data_and_meta()
    #
    # # hourly data
    KinergyDataLoader(DATA_DIR / "kinergy", res="hourly").write_data_and_meta()
    DHDataLoader(DATA_DIR / "district_heating_data", res="hourly").write_data_and_meta()

    logger.info("Finish data loading")

    ds = Dataset()
    ds.create_and_clean()

    ds_hourly = Dataset(res="hourly")
    ds_hourly.create_and_clean()
