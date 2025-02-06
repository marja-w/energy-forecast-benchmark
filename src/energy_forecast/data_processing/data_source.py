import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

from src.energy_forecast.config import RAW_DATA_DIR, DATA_DIR, PROCESSED_DATA_DIR, REFERENCES_DIR
from src.energy_forecast.data_processing.transform import update_df_with_corrections
from src.energy_forecast.util import sum_columns, replace_title_values, remove_leading_zeros


class DataLoader(object):  #
    def __init__(self, input_path: Path):
        self.input_path = input_path  # path to input data file or folder
        self.df_raw = pl.DataFrame()  # raw data
        self.df_t = pl.DataFrame()  # transformed data
        self.res = ""  # resolution of transformed data, either daily or hourly

        self.meta_cols = list()  # list of columns for creating meta data file
        self.name = ""  # name for the given data source

    def load(self):
        """
        Load data from raw data directory. Remove corrupt values, but leave structure. Store DataFrame.
        :return:
        """
        pass

    def transform(self, res: str):
        """
        Bring data into right data structure.
        :return:
        """
        logger.info(f"Transforming data to {res} format")
        self.res = res
        self.df_raw = self.df_raw.with_columns(pl.lit(self.name).alias("source"))

    def save(self):
        if self.df_t.is_empty():
            output_path = RAW_DATA_DIR / f"{self.name}_raw.csv"
            self.df_raw.write_csv(output_path)
            logger.success(f"Raw data saved to {output_path}")
        else:
            output_path = RAW_DATA_DIR / f"{self.name}_{self.res}.csv"
            self.df_t.write_csv(output_path)
            logger.success(f"Transformed data ({self.res}) saved to {output_path}")

    def write_meta_data(self):
        # extract metadata
        df_meta = self.df_t.group_by(self.meta_cols).agg()
        meta_data_csv = RAW_DATA_DIR / f"{self.name}_meta.csv"
        logger.info(f"Writing meta data file to {meta_data_csv}")
        df_meta.write_csv(meta_data_csv)


class LegacyDataLoader(DataLoader):
    def __init__(self, input_path: Path):
        super().__init__(input_path)
        self.meta_cols = ["id", "GSM_ID", "TP", "Tag", "Title", "Type", "s", "m", "CircuitType", "CircuitNum",
                          "co2koeffizient"]
        self.name = "legacy"

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
        self.df_raw = df

    def transform(self, res: str = "daily"):
        super().transform(res=res)
        df = (self.df_raw.with_columns(
            pl.col("ArrivalTime").str.to_date().dt.date().alias("date"),
            pl.col("lastAggregated").str.to_datetime(),
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
            pl.col("Val").diff().over(pl.col("id")).alias("diff")
            # compute diff column  TODO: Datenlücken bei date? check
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
        self.df_t = df


class KinergyDataLoader(DataLoader):
    def __init__(self, input_path: Path):
        super().__init__(input_path)
        self.meta_cols = ["id", "renewable_energy_used", "has_pwh", "pwh_type", "building_type", "orga", "complexity",
                          "complexity_score", "env_id"]
        self.name = "kinergy"

    def load(self):
        logger.info("Loading kinergy data")
        df = pl.DataFrame()

        logger.info("Loading meta data file")
        meta_data_file = self.input_path / "kinergy_eco_u_list.json"
        with open(meta_data_file, "r", encoding="UTF-8") as f:
            meta_data = json.loads(f.read())

        for eco_u_id in list(meta_data.keys()):
            print(f"EcoU {eco_u_id} - {meta_data[eco_u_id]['name']}")
            sensor_file = f"{self.input_path}/consumption_data/{eco_u_id}_consumption.csv"
            if os.path.isfile(sensor_file):
                # read the data
                df_s = pl.read_csv(sensor_file, null_values="null")
                df_s = df_s.with_columns(pl.col("bucket").str.to_datetime().alias("datetime"),
                                         ).select(["datetime", "sum_kwh", "sum_kwh_diff", "env_temp"])
                # remove zeros that are recorded before the sensor was actually working
                df_s = remove_leading_zeros(df_s)
                df_s = df_s.with_columns(pl.lit(eco_u_id).alias("id"))
                df_s = df_s.cast({"env_temp": pl.Float64})  # if temp is null
                logger.info(f"Adding {len(df_s)} datapoints for {eco_u_id}")
                df = pl.concat([df, df_s])
            else:
                logger.warning(f"Missing file for: {eco_u_id}")

        df = df.sort(["id", "datetime"])
        logger.success(f"Raw data length: {len(df)}")
        self.df_raw = df

    def transform(self, res: str):
        super().transform(res=res)
        if res == "daily":
            # aggregate values to daily values
            time_delta = "1d"
        elif res == "hourly":
            time_delta = "1h"
        else:
            raise ValueError(f"Unknown res type: {res}")
        df = self.df_raw.group_by_dynamic(
            index_column="datetime", every=time_delta, group_by=["id"]
        ).agg([
            pl.col("env_temp").mean().alias("avg_env_temp"),
            pl.col("sum_kwh").max().alias("value"),
            pl.col("sum_kwh_diff").sum().alias("diff")
        ]).sort(["id", "datetime"])

        # remove duplicates
        df = df.unique(subset=["id", "datetime"]).sort(["id", "datetime"])

        if res == "daily":
            df = df.with_columns(pl.col("datetime").dt.date().alias("date")
                                 ).select(["id", "date", "value", "diff", "avg_env_temp"])

        logger.success(f"Data length after transforming: {len(df)}")
        self.df_t = df


class DHDataLoader(DataLoader):
    def __init__(self, input_path: Path):
        super().__init__(input_path)
        self.name = "dh"
        self.main_counter = pl.DataFrame()

    def load(self):
        logger.info("Loading DH data")
        dh_data_folder_first_part = self.input_path / "2024_01_29 Projekt KI-FW Data Export (2022_08-2024_01_29)" / "data"
        dh_data_folder_second_part = self.input_path / "2024_05_14 Projekt KI-FW Data Export (2024_02_08-2024_05_14)"
        file_path_dump_1 = dh_data_folder_first_part / "dump.csv"
        file_path_dump_2 = dh_data_folder_second_part / "dump.csv"
        dh_data_folder = self.input_path / "data"

        df = pl.scan_csv([file_path_dump_1, file_path_dump_2]).collect()
        logger.info(f"Raw data length: {len(df)}")
        logger.info("Filtering data for FW Wärmemenge endpoints")

        # filter for data endpoints of type "FW Wärmemenge" and store one .csv for each sensor
        sensors = list()
        eco_u_data_merge = dict()
        for file in os.listdir(dh_data_folder_first_part):
            filename = os.fsdecode(file)
            filename, file_extension = os.path.splitext(filename)
            if file_extension == ".json":
                with open(dh_data_folder_first_part / file, "r", encoding="UTF-8") as f:
                    eco_u_data = json.loads(f.read())

                # collect meta data
                eco_u_data_point = {filename: eco_u_data["economic_unit"]}
                eco_u_data_merge.update(eco_u_data_point)

                # filter for wärmemenge
                for data_point in eco_u_data["datapoint_data"]:
                    if data_point["title"] == "FW Wärmemenge":  # filter FW Wärmemenge
                        sensor_id = data_point["id"]
                eco_u_id, data_provider_id = filename.split(".")
                df_sensor = df.filter(pl.col("eco_u_id") == eco_u_id,
                                      pl.col("data_provider_id") == data_provider_id,
                                      pl.col("sensor_id") == sensor_id
                                      ).sort(pl.col("time"))
                # print(f"Adding {len(df_sensor)} for {eco_u_id}")
                sensors.append(df_sensor)
                # store_csv_path = dh_data_folder / f"{filename}.csv"
                # df_sensor.write_csv(store_csv_path)
                # print("Created " + str(store_csv_path))
        # write meta data json
        json_object = json.dumps(eco_u_data_merge)
        logger.info(f"Writing meta data .json to {dh_data_folder / 'eco_u_ids.json'}")
        with open(dh_data_folder / "eco_u_ids.json", "w") as outfile:
            outfile.write(json_object)

        # merge filtered sensors to one dataframe
        df = pl.concat(sensors)

        # filter data for main counter of the buildings, where the value is the largest of the building
        self.main_counter = df.group_by(
            pl.col(["eco_u_id", "data_provider_id"])
        ).agg(pl.len(),
              pl.col("time").min().alias("min_time"),
              pl.col("time").max().alias("max_time"),
              pl.col("value").max().alias("max_value")
              ).group_by(
            "eco_u_id").agg(
            pl.col("data_provider_id").filter(pl.col("max_value") == pl.col("max_value").max())).with_columns(
            pl.col("data_provider_id").map_elements(lambda v: v[0], return_dtype=pl.String)
        )

        df = df.filter(
            pl.struct("eco_u_id", "data_provider_id").is_in(self.main_counter)  # filter only main counters
        )
        logger.info(f"Data length after filtering: {len(df)}")
        self.df_raw = df

    def transform(self, res: str):
        super().transform(res=res)
        if res == "daily":
            time_delta = "1d"
        elif res == "hourly":
            time_delta = "1h"
        else:
            raise ValueError(f"Unknown res type: {res}")
        df = self.df_raw.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ").alias("datetime"))
        df = df.group_by_dynamic(index_column="datetime", every=time_delta,
                                 group_by=["sensor_id", "eco_u_id", "data_provider_id"]
                                 ).agg(pl.col("value").max())

        df = df.rename({"eco_u_id": "id"}).select(["id", "datetime", "value", "sensor_id"]
                                                  ).unique(subset=["id", "datetime"]
                                                           ).sort(["id", "datetime"])
        # compute diff column
        df = df.with_columns(pl.col("value").diff().over(pl.col("id")).alias("diff")).drop_nulls(subset=["diff"])
        logger.info(f"Data length after transforming: {len(df)}")
        self.df_t = df

    def write_meta_data(self):
        logger.info("Writing meta data file")
        eco_u_data_file = DATA_DIR / "district_heating_data" / "eco_u_ids.json"
        with open(eco_u_data_file, "r", encoding="UTF-8") as f:
            meta = json.loads(f.read())
        meta_data = list()
        for key, item in meta.items():
            eco_u_id, data_provider_id = key.split(".")
            address = item["address"]["street_address"]
            city = item["address"]["address_locality"]
            postal_code = item["address"]["postal_code"]
            country = item["address"]["address_country"]
            meta_data.append(
                {"eco_u_id": eco_u_id, "data_provider_id": data_provider_id, "address": address, "city": city,
                 "postal_code": postal_code, "country": country})
        df_meta = pl.DataFrame(meta_data).filter(pl.struct("eco_u_id", "data_provider_id").is_in(self.main_counter))
        df_meta = df_meta.with_columns(pl.lit("district heating").alias("primary_energy"),
                                       pl.lit("kwh").alias("unit_code"))
        logger.success(f"Writing meta data file to {RAW_DATA_DIR / f'{self.name}_meta.csv'}")
        df_meta.write_csv(RAW_DATA_DIR / f"{self.name}_meta.csv")
