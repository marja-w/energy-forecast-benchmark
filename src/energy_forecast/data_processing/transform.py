from datetime import datetime
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from src.energy_forecast.config import PROCESSED_DATA_DIR, REFERENCES_DIR
from src.energy_forecast.util import remove_vals_by_title, subtract_diff_data, replace_title_values

app = typer.Typer()


def create_new_id(df: pl.DataFrame):
    """
    Create new ID as combination of GSM_ID, TP, and Tag. Filter only for Gas values.
    :param df:
    :return:
    """
    return df.with_columns(pl.when(
        pl.col("TP") == "Gas",
        pl.col("Tag") == "VG"
    ).then(
        pl.col("GSM_ID").cast(pl.String).add("GVG")
    ).when(
        pl.col("TP") == "Gas",
        pl.col("Tag") == "VA"
    ).then(
        pl.col("GSM_ID").cast(pl.String).add("GVA")
    ).alias("new_id")).filter(
        pl.col("TP") == "Gas"  # only examples with TP="Gas" are relevant
    )


def create_date(df: pl.DataFrame):
    """
    Create new data column from string
    :param df:
    :return:
    """
    return df.with_columns(pl.col("ArrivalTime").dt.date().alias('date'))  # add date column


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

    logger.info(f"Filtered {filtered_count} rows for column {column} for ID {df['new_id'][0]}")

    return filtered_df


def filter_outliers_by_id(df, filter_column):
    """
    Apply the filter_outliers_iqr function to each subset of the DataFrame grouped by the id_column.

    :param df: polars DataFrame
    :param filter_column: column name to filter outliers
    :return: DataFrame with outliers removed for each group
    """

    filtered_df = df.group_by("new_id").map_groups(lambda group: filter_outliers_iqr(group, filter_column))
    return filtered_df


def update_df_with_corrections(df: pl.DataFrame, correction_csv_path: Path) -> pl.DataFrame:
    """
    Updates the main DataFrame `df` with corrections from a CSV file.

    Parameters:
    - df: polars.DataFrame
        The main DataFrame containing energy consumption data.
    - correction_csv_path: str
        The file path to the correction CSV.

    Returns:
    - polars.DataFrame
        The updated DataFrame with corrected `qmbehfl` and `anzlwhg` values.
    """

    # Step 1: Read the correction CSV
    correction = pl.read_csv(
        correction_csv_path,
        separator=",",
        has_header=True,
        try_parse_dates=False,  # Prevent automatic date parsing
        # Specify encoding if necessary, e.g., encoding="utf8"
    )

    # Step 2a: Drop entries where qmbehfl is '?'
    correction = correction.filter(pl.col('qmbehfl') != '?')

    # Step 2b: Replace '?' in anzlwhg with 0
    correction = correction.with_columns([
        pl.when(pl.col('anzlwhg') == '?')
        .then(0)
        .otherwise(pl.col('anzlwhg'))
        .cast(pl.Int64)
        .alias('anzlwhg')
    ])

    # Step 2c: Replace ',' with '.' in qmbehfl and convert to float
    correction = correction.with_columns([
        pl.col('qmbehfl')
        .str.replace(",", ".")
        .cast(pl.Float64)
        .alias('qmbehfl')
    ])

    # Step 3: Select relevant columns for merging
    correction = correction.select(['new_id', 'qmbehfl', 'anzlwhg'])

    # Step 4: Join the correction data with the main DataFrame on 'new_id'
    # Perform a left join to retain all rows from df
    df_updated = df.join(
        correction,
        on='new_id',
        how='left',
        suffix='_corr'
    )

    # Step 5: Update 'qmbehfl' and 'anzlwhg' with corrected values where available
    df_updated = df_updated.with_columns([
        pl.when(pl.col('qmbehfl_corr').is_not_null())
        .then(pl.col('qmbehfl_corr'))
        .otherwise(pl.col('qmbehfl'))
        .alias('qmbehfl'),

        pl.when(pl.col('anzlwhg_corr').is_not_null())
        .then(pl.col('anzlwhg_corr'))
        .otherwise(pl.col('anzlwhg'))
        .alias('anzlwhg')
    ])

    # Step 6: Drop the temporary correction columns
    df_updated = df_updated.drop(['qmbehfl_corr', 'anzlwhg_corr'])

    return df_updated


@app.command()
def main(
        input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        output_path: Path = PROCESSED_DATA_DIR / "transform.csv",
):
    df = pl.read_csv(input_path)  # Lade die CSV-Datei in ein Polars DataFrame
    df = df.cast({"ArrivalTime": pl.Datetime})
    logger.info(f"Current length of data: {len(df)}")

    logger.info("Processing dataset...")

    df = create_new_id(df)  # add new_id and filter for gas values only
    df = create_date(df)

    logger.info("Cleaning dataset...")
    # subtract data from Schenfelder Holt 135 BHKW from Gesamt and replace the values
    df = subtract_diff_data(df, "400768GVG", "400768GVA")

    # remove BHKWs and GAPWs from dataset
    df = remove_vals_by_title(df, ["Gaszähler BHKW", "Gaszähler GAWP"])

    # unify naming of gas meters
    df = replace_title_values(df, [("Gas Zähler", "Gaszähler"),
                                   ("Gaszähler Z Kessel", "Gaszähler Kessel Z"),
                                   ("Gas", "Gaszähler"),
                                   ("Gesamt Gaszähler", "Gaszähler Gesamt")])
    logger.info(f"Current length of data: {len(df)} after removing BHKWs and GAPWs")

    # add missing qmbehfl and anzlwhg values from correction file
    correction_csv_path = REFERENCES_DIR / "liegenschaften_missing_qm_wohnung.csv"
    df = update_df_with_corrections(df, correction_csv_path)
    logger.info(f"Current length of data: {len(df)} after adding missing qmbehfl and anzlwhg values")

    # manually remove corrupt days for Wilhelmstraße 33-41
    df = df.filter(~((pl.col("new_id") == "400305GVG") & (
        pl.col("date").is_between(datetime(2019, 11, 6), datetime(2019, 12, 21)))))  # remove corrupt days
    # manually remove corrupt days for Kaltenbergen 22
    df = df.filter(~((pl.col("new_id") == "400204GVA") & (
        pl.col("date").is_between(datetime(2020, 3, 12), datetime(2020, 4, 28)))))
    # manually remove corrupt days for Dahlgrünring 5-9
    df = df.filter(~((pl.col("new_id") == "400711GVG") & (
        pl.col("date").is_between(datetime(2021, 4, 8), datetime(2021, 6, 28)))))

    df = remove_neg_diff_vals(df)
    logger.info(f"Current length of data: {len(df)} after removing negative diffs")

    df = filter_outliers_by_id(df, "diff")
    logger.info(f"Current length of data: {len(df)} after filtering outliers")
    print(f"Current length of data: {len(df)}")

    logger.success("Processing dataset complete.")
    df.write_csv(output_path)


if __name__ == "__main__":
    app()
