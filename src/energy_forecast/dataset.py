import json
import os
from typing import Any

import pandas as pd
import polars as pl
import typer
from loguru import logger

from config import RAW_DATA_DIR, DATA_DIR
from src.energy_forecast.config import REPORTS_DIR
from src.energy_forecast.data_processing.data_source import LegacyDataLoader, KinergyDataLoader, DHDataLoader


if __name__ == '__main__':
    # DATA LOADING
    logger.info("Start data loading")

    # daily data
    LegacyDataLoader(DATA_DIR / "legacy_data" / "legacy_systen_counter_daily_values.csv").write_data_and_meta()
    KinergyDataLoader(DATA_DIR / "kinergy").write_data_and_meta()
    DHDataLoader(DATA_DIR / "district_heating_data").write_data_and_meta()

    # hourly data
    KinergyDataLoader(DATA_DIR / "kinergy", res="hourly").write_data_and_meta()
    DHDataLoader(DATA_DIR / "district_heating_data", res="hourly").write_data_and_meta()

    logger.info("Finish data loading")

