import os
from datetime import timedelta

import darts
import polars as pl
from darts.utils.missing_values import fill_missing_values, extract_subseries
from loguru import logger
from matplotlib import pyplot as plt

from src.energy_forecast.config import FIGURES_DIR, MIN_GAP_SIZE_DAILY
from src.energy_forecast.plots import plot_dataframe, plot_filtered_data_points
from src.energy_forecast.utils.util import get_missing_dates, find_time_spans


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


def remove_positive_jumps(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("diff") < pl.col("diff").median() * 50).over("id"))


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

    # plot_filtered_data_points(df, column, filtered_df)

    # logger.info(f"Filtered {filtered_count} rows for column {column} for ID {df['id'][0]}")

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
    # logger.info(f"Filtered {starting_n - len(df)} rows for ID {df['id'][0]}")
    return df


def filter_connection_errors(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    start_n = len(df)
    df = df.with_row_index()
    df = df.with_columns(pl.col("datetime").str.to_datetime())
    m_dates = get_missing_dates(df, frequency=freq, store=False)["missing_dates"].to_list()[0]
    delta = timedelta(days=1) if freq == "D" else timedelta(hours=1)
    spans = find_time_spans(m_dates, delta)

    if freq == "h":
        spans = spans.filter(pl.col("n") > 2)  # only remove if span big enough

    for row in spans.iter_rows():
        start = row[0]
        end = row[1]
        n = row[2]
        error_val_idx = df.filter(pl.col("datetime") == end + delta)["index"].item()
        df = df.filter(~(pl.col("index") == error_val_idx))  # remove row with erroneous value

    df = df.drop(["index"])
    # logger.info(f"Filtered {start_n - len(df)} rows for ID {df['id'][0]}")
    return df


def filter_flat_lines_by_id(df: pl.DataFrame, thresh: int) -> pl.DataFrame:
    filtered_df = df.group_by("id").map_groups(lambda group: filter_flat_lines(group, thresh))
    return filtered_df


def filter_connection_errors_by_id(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    filtered_df = df.group_by("id").map_groups(lambda group: filter_connection_errors(group, freq))
    return filtered_df


def interpolate_values(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    data_source = df["source"].mode().item()
    building_id = df["id"].mode().item()
    # use darts and pandas for interpolating values
    series = darts.TimeSeries.from_dataframe(df, time_col="datetime", value_cols="value", freq=freq,
                                             fill_missing_dates=True)  # fill missing dates with nan values
    series_filled = fill_missing_values(series, "auto",
                                        method="linear")  # use pandas interpolate for linear interpolation
    df_filled = series_filled.to_dataframe(backend="polars", time_as_index=False)  # convert back to polars dataframe
    df_filled = df_filled.with_columns(pl.col("value").diff().alias("diff"),  # recompute diff column with added values
                                       pl.lit(data_source).alias("source"),
                                       pl.lit(building_id).alias("id"))
    if df_filled["datetime"][0] == df["datetime"][0]:
        df_filled = df_filled.fill_null(df["diff"][0])  # replace first diff with known diff from old df
    else:
        df_filled = df_filled.drop_nans(subset=["diff"])
    # plot_interpolated_series(series_filled, building_id, data_source)
    return df_filled


def interpolate_values_by_id(df: pl.DataFrame, freq: str = "D"):
    interpolated_df = df.group_by("id").map_groups(lambda group: interpolate_values(group, freq))
    return interpolated_df


def split_series(df: pl.DataFrame, min_gap_size: int, res: str, plot: bool = True) -> pl.DataFrame:
    freq = "D" if res == "daily" else "h"
    data_source = df["source"].mode().item()
    building_id = df["id"].mode().item()
    # use darts for extracting subseries
    series = darts.TimeSeries.from_dataframe(df, time_col="datetime", value_cols=["value", "diff"], freq=freq,
                                             fill_missing_dates=True)
    # if plot:
    #     series.plot(label=building_id)
    #     plt.show()
    subseries = extract_subseries(series, min_gap_size=min_gap_size)

    # create output directory if it doesnt exist yet
    # delete all files in output directory if it exists
    output_directory = FIGURES_DIR / "interpolated_and_split_data" / freq
    if os.path.exists(output_directory):
        filelist = [f for f in os.listdir(output_directory) if f.endswith(".csv")]
        for f in filelist:
            os.remove(os.path.join(output_directory, f))
    else:
        os.makedirs(output_directory)

    if len(subseries) == 1:
        if plot:
            plot_dataframe(df, building_id, data_source, folder=output_directory)
        return df  # if we dont have any gaps, return original dataframe
    # if plot:
    #     [s["diff"].plot(label=building_id) for s in subseries]
    #     plt.show()
    df_subs_raw = [s.to_dataframe(backend="polars", time_as_index=False) for s in subseries]
    df_subs = list()
    for idx, df_sub in enumerate(df_subs_raw):
        if len(df_sub) > min_gap_size:  # dont add if it is shorter than min gap size
            new_building_id = f"{building_id}-{idx}"
            df_sub = df_sub.with_columns(pl.lit(data_source).alias("source"),
                                         pl.lit(new_building_id).alias("id"))
            if plot: plot_dataframe(df_sub, new_building_id, data_source,
                                    folder=output_directory)
            df_subs.append(df_sub)
    if len(df_subs) > 0:
        df_concat = pl.concat(df_subs)
        df_concat = df_concat.with_columns(pl.col("datetime").dt.cast_time_unit("ns"))
        return df_concat
    else:
        return pl.DataFrame(schema={"datetime": pl.Datetime, "value": pl.Float64, "diff": pl.Float64, "id": pl.String,
                                    "source": pl.String})


def split_series_by_id_list(df: pl.DataFrame, min_gap_size: int, res: str, plot: bool = False) -> pl.DataFrame:
    concat_df = pl.DataFrame(
        schema={"datetime": pl.Datetime, "value": pl.Float64, "diff": pl.Float64, "id": pl.String, "source": pl.String})
    concat_df = concat_df.with_columns(pl.col("datetime").dt.cast_time_unit("ns"))
    for b_idx in df["id"].unique():
        df_b = df.filter(pl.col("id") == b_idx)
        df_b = split_series(df_b, min_gap_size=min_gap_size, res=res, plot=plot)
        concat_df = pl.concat([concat_df, df_b], how="diagonal")
    return concat_df


def split_series_by_id(df: pl.DataFrame, min_gap_size: int, res: str, plot: bool = False) -> pl.DataFrame:
    interpolated_df = df.group_by("id").map_groups(lambda group: split_series(group, min_gap_size, res, plot))
    return interpolated_df


def remove_null_series(df: pl.DataFrame, features: list[str]) -> bool:
    df_bool = df[features].with_columns(pl.all().is_null())
    for col_name in features:
        if df_bool[col_name].all():
            return True
    return False


def remove_null_series_by_id(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    idxs_to_remove = df[["id"] + features].group_by("id").agg(pl.all().is_null().all()).with_columns(
        any=pl.any_horizontal(features)).filter(pl.col("any"))["id"].to_list()  # all ids where at least one column only contains nulls
    filtered_df = df.filter(~(pl.col("id").is_in(idxs_to_remove)))
    return filtered_df
