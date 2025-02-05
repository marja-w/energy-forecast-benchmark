from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from src.energy_forecast.config import RAW_DATA_DIR, DATA_DIR, PROCESSED_DATA_DIR, REFERENCES_DIR
from src.energy_forecast.data_processing.transform import update_df_with_corrections
from src.energy_forecast.util import sum_columns, replace_title_values


class DataSource(object):  #
    def __init__(self, input_path: Path):
        self.input_path = input_path  # path to input data file or folder
        self.df = pl.DataFrame()

    def load(self):
        """
        Load data from raw data directory. Remove corrupt values, but leave structure. Store DataFrame.
        :return:
        """
        pass

    def transform(self):
        """
        Bring data into right data structure.
        :return:
        """
        pass

    def save(self, output_path: Path):
        self.df.write_csv(output_path)
        logger.success(f"Data saved to {output_path}")


class LegacyDataSource(DataSource):
    def __init__(self, input_path: Path):
        super().__init__(input_path)
        self.output_path = RAW_DATA_DIR / "legacy_daily.csv"

    def load(self):
        logger.info("Loading legacy data")
        df = pl.read_csv(self.input_path, schema_overrides={"plz": pl.String, "m": pl.Float64})
        df = df.with_columns(pl.col("ArrivalTime").str.to_datetime().alias("date"))
        logger.info("Raw data length: {}".format(len(df)))

        logger.info("Filtering data for gas values")
        # tps = ["GAS", "Gas1", "Gas_Gesamt", "GasKessel", "PulseCount0", """Gas""", """gas"""]
        # df = df.filter((pl.col("CircuitPoint") == "GAS").or_(pl.col("TP").is_in(tps)))  # TODO: wait for input from timo
        df = df.filter((pl.col("CircuitPoint") == "GAS"))

        logger.info("Removing corrupt data loggers")
        # remove Schenfelder Holt BHWK (Unterzähler)
        df = df.filter(~((pl.col("Title") == "Gaszähler BHKW") & (pl.col("adresse") == "Schenfelder Holt 135")))

        # remove for now TODO: update with info from timo
        df = df.filter(~((pl.col("adresse") == "Chemnitzstraße 25c") & (
                pl.col("Title") == "Gaszähler GWP")))  # Hauptzähler, aber auch Solarthermie
        df = df.filter(~(pl.col("adresse") == "Heidrehmen 1"))  # Hauptzähler fehlt

        # manually remove corrupt days
        df = df.filter(~((pl.col("adresse") == "Wilhelmstraße 33-41") & (
            pl.col("date").is_between(datetime(2019, 11, 6), datetime(2019, 12, 21)))))
        df = df.filter(~((pl.col("adresse") == "Dahlgrünring 5-9") & (
            pl.col("date").is_between(datetime(2021, 4, 8), datetime(2021, 6, 28)))))
        df = df.filter(~((pl.col("adresse") == "Kaltenbergen 22") & (
            pl.col("date").is_between(datetime(2020, 3, 12), datetime(2020, 4, 28)))))

        logger.info("Merge split data loggers")
        # sum dataloggers to get overall consumption
        df = sum_columns(df, "Brandenbaumer Landstraße 177", "Gaszähler Kessel", "Gaszähler GAWP")
        df = sum_columns(df, "Frankfurter Straße 29 ", "Gaszähler Kessel H29", "Gaszähler Kessel H17")
        df = sum_columns(df, "Iserbrooker Weg 72", "Gaszähler Kessel ", "Gaszähler BHKW")
        df = sum_columns(df, "Martinistraße 44", "GAWP 1 Gaszähler", "GAWP 2 Gaszähler")
        df = sum_columns(df, "Op´n Hainholt 4-18", "Gaszähler Kessel", "Gaszähler BHKW")
        df = sum_columns(df, "Sven Hedin Str.11", "Gaszähler Kessel", "Gaszähler BHKW")

        logger.success("Data length now: {}".format(len(df)))
        self.df = df

    def transform(self):
        logger.info("Transforming data")
        df = (self.df.with_columns(
            pl.col("ArrivalTime").str.to_date().dt.date().alias("date"),
            pl.col("lastAggregated").str.to_datetime(),
            pl.lit("legacy").alias("source"),
            pl.concat_str(  # create id
                [pl.col("GSM_ID"),
                 pl.col("TP").str.slice(0, 1),
                 pl.col("Tag")]
            ).alias("id"),
        ).unique(subset=["GSM_ID", "TP", "Tag", "date"]  # remove duplicates
                 ).rename({"CircuitPoint": "primary_energy"}
                          ).sort(by=["GSM_ID", "TP", "Tag", "date"])
              .with_columns(
            pl.col("primary_energy").str.to_lowercase(),
            pl.when(pl.col("Unit") == "CBM")
            .then(pl.col("Val") * 11.3).alias("Val"),
            pl.when(pl.col("Unit") == "CBM")
            .then(pl.lit("KWH")).alias("Unit"),
            pl.col("plz").str.strip_chars().cast(pl.Int64),
            pl.col("Val").diff().over(pl.col("id")).alias("diff")  # compute diff column  TODO: Datenlücken bei date? check
        ).drop_nulls(subset=["diff"])
              )

        ## TODO: move to dataset later
        # unify naming of gas meters
        df = replace_title_values(df, [("Gas Zähler", "Gaszähler"),
                                       ("Gaszähler Z Kessel", "Gaszähler Kessel Z"),
                                       ("Gas", "Gaszähler"),
                                       ("Gesamt Gaszähler", "Gaszähler Gesamt")])

        # add missing qmbehfl and anzlwhg values from correction file
        correction_csv_path = REFERENCES_DIR / "liegenschaften_missing_qm_wohnung.csv"
        df = update_df_with_corrections(df, correction_csv_path)

        logger.info(f"Current length of data: {len(df)} after transforming")
        logger.success("Transforming legacy dataset complete.")
        self.df = df

    def write_meta_data(self):
        # extract metadata
        df_meta = self.df.group_by(pl.col("id")).agg(pl.col("GSM_ID").max(), pl.col("TP").max(), pl.col("Tag").max(),
                                                pl.col("Title").max(), pl.col("Type").max(), pl.col("s").max(),
                                                pl.col("m").max(), pl.col("CircuitType").max(),
                                                pl.col("CircuitNum").max(), pl.col("co2koeffizient").max(),
                                                pl.col("Objekttitel").max())
        meta_data_csv = PROCESSED_DATA_DIR / "legacy_meta.csv"
        logger.info(f"Writing meta data file to {meta_data_csv}")
        df_meta.write_csv(meta_data_csv)


if __name__ == '__main__':
    ds = LegacyDataSource(DATA_DIR / "legacy_data" / "legacy_systen_counter_daily_values.csv")
    ds.load()
    ds.transform()
    ds.save(RAW_DATA_DIR / "legacy_daily.csv")
