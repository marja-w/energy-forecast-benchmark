from loguru import logger
import os

import polars as pl
import pgeocode
from meteostat import Point, Daily, Hourly

import holidays

from src.energy_forecast.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR, FEATURES_DIR, META_DIR

# INPUT PATHS
dataset_daily_csv_ = PROCESSED_DATA_DIR / "dataset_daily.csv"

# OUTPUT PATHS
city_df_path = FEATURES_DIR / "city_info.csv"
output_path_weather_daily = FEATURES_DIR / "weather_daily.csv"
output_path_weather_hourly = FEATURES_DIR / "weather_hourly.csv"


def create_data_df():
    """
    Creates a DataFrame by reading a CSV file and processing it to include
    additional metadata. The method ensures the inclusion of time intervals
    for every city based on postal codes derived from various metadata files.

    :return: A processed Polars DataFrame with appended metadata columns
    :rtype: pl.DataFrame
    """
    data_df = pl.read_csv(dataset_daily_csv_).with_columns(pl.col("datetime").str.to_datetime())
    # Find time intervals for every city
    data_df = data_df.with_columns(
        pl.coalesce(data_df.join(pl.read_csv(META_DIR / "kinergy_meta.csv"), on="id", how="left")["plz"],
                    data_df.join(pl.read_csv(META_DIR / "legacy_meta.csv"), on="id", how="left")["plz"],
                    data_df.join(pl.read_csv(META_DIR / "dh_meta.csv").rename({"postal_code": "plz"}),
                                 on="id", how="left")["plz"],
                    ).str.strip_chars())
    return data_df


def generate_weather_dfs():
    """
    Generate Weather DataFrames from a given daily dataset, metadata files, and external APIs.

    This function processes weather data by combining information from the daily weather dataset, metadata on cities
    and postal codes, and external APIs (Meteostat and Postcode Geocoding). It calculates weather statistics for every
    city within the constraints of date ranges and geographic locations, including daily and hourly weather data.

    It performs the following key tasks:

    1. Reads daily weather data, converting date-time strings to proper datetime objects.
    2. Links metadata on postal codes to integrate additional city-related information.
    3. Calculates time intervals for weather data availability for each city.
    4. Maps postal codes to geographic coordinates and administrative divisions.
    5. Fetches daily weather data from the Meteostat API for specified postal codes and intervals.
    6. Enhances daily weather data by fetching hourly humidity information and merging it back with the daily dataset.
    7. Exports preprocessed daily and hourly weather data into separate CSV files.

    **Note:** The data includes weather information such as temperature, wind speed, air pressure, and humidity
    (retrieved separately). It accounts for variability in weather data availability across cities.

    :raises FileNotFoundError: If required files (e.g., daily dataset or metadata files) are not found.
    :raises HTTPError: If there are issues with API responses during data fetching.
    :raises ValueError: If invalid data or parameters are encountered during processing.

    :param None: This function does not accept any parameters.
    :returns: None. The processed weather data is written directly to CSV files at predefined locations.
    """
    data_df = create_data_df()
    city_df = data_df.group_by(pl.col("plz")).agg(pl.col("datetime").min().alias("min_date"),
                                                  pl.col("datetime").max().alias("max_date")).filter(
        ~(pl.col("plz") == "2700"))  # wien
    # Add coordinates to every city
    rows = list()
    for plz in city_df["plz"].unique():
        data = pgeocode.Nominatim("de").query_postal_code(str(plz))
        rows.append({"plz": plz, "lat": data["latitude"], "lon": data["longitude"], "state": data["state_code"]})
    info_df = pl.DataFrame(rows)
    city_df = city_df.join(info_df, on="plz", how="left")
    city_df.write_csv(city_df_path)
    weather_dfs = list()
    for row in city_df.iter_rows():
        start = row[1]
        end = row[2]
        loc = Point(row[3], row[4])

        data = Daily(loc, start, end)
        data = data.fetch()
        weather_dfs.append(pl.from_pandas(data.reset_index()).with_columns(pl.lit(row[0]).alias("plz")))
    weather_df = pl.concat(weather_dfs)
    # From the [meteostat](https://dev.meteostat.net/python/daily.html#api) documentation:
    #
    # Column	Description	Type
    #
    # station	The Meteostat ID of the weather station (only if query refers to multiple stations)	String
    #
    # time	The date	Datetime64
    #
    # tavg	The average air temperature in °C	Float64
    #
    # tmin	The minimum air temperature in °C	Float64
    #
    # tmax	The maximum air temperature in °C	Float64
    #
    # prcp	The daily precipitation total in mm	Float64
    #
    # snow	The snow depth in mm	Float64
    #
    # wdir	The average wind direction in degrees (°)	Float64
    #
    # wspd	The average wind speed in km/h	Float64
    #
    # wpgt	The peak wind gust in km/h	Float64
    #
    # pres	The average sea-level air pressure in hPa	Float64
    #
    # tsun	The daily sunshine total in minutes (m)	Float64
    #
    # Humidity is missing from Daily-data, we can retrieve hourly data and merge to daily data
    weather_dfs = list()
    for row in city_df.iter_rows():
        start = row[1]
        end = row[2]
        loc = Point(row[3], row[4])

        data = Hourly(loc, start, end)
        data = data.fetch()
        weather_dfs.append(pl.from_pandas(data.reset_index()).group_by_dynamic(
            index_column="time", every="1d"
        ).agg(pl.col("rhum").mean().alias("hum_avg"),
              pl.col("rhum").min().alias("hum_min"),
              pl.col("rhum").max().alias("hum_max")
              ).with_columns(pl.lit(row[0]).alias("plz")))
    weather_df_hourly = pl.concat(weather_dfs)
    # Add to other weather data
    weather_df = weather_df_hourly.join(weather_df, on=["plz", 'time'], how="left")
    weather_df.write_csv(output_path_weather_daily)
    # Get hourly weather data as well
    weather_dfs = list()
    for row in city_df.iter_rows():
        start = row[1]
        end = row[2]
        loc = Point(row[3], row[4])

        data = Hourly(loc, start, end)
        data = data.fetch()
        weather_dfs.append(pl.from_pandas(data.reset_index()).with_columns(pl.lit(row[0]).alias("plz")))
    weather_df_hourly = pl.concat(weather_dfs)
    weather_df_hourly.write_csv(output_path_weather_hourly)
    logger.info(f"Weather Data written to {output_path_weather_daily} and {output_path_weather_hourly}")


def generate_holiday_df():
    """
    Generates holiday DataFrames for both German national and state-wide holidays, and concatenates
    these holiday records with an existing holiday dataset. The process involves fetching holiday
    data for each state from a predefined range of years, restructuring it for consistency, and
    then saving the combined dataset as a CSV file.

    :return: None
    """
    holiday_dict = dict()
    ger_holidays = holidays.country_holidays("DE", years=range(2018, 2024))
    holiday_dict.update(ger_holidays)
    holidays_state_dict = dict()
    city_df = pl.read_csv(city_df_path)
    for state in city_df["state"].unique():
        state_holidays = holidays.country_holidays("DE", subdiv=state, years=range(2018, 2024))
        holidays_state_dict.update({state: state_holidays})
    holiday_list = list()
    for state in city_df["state"].unique():
        for date, holiday in holidays_state_dict[state].items():
            holiday_list.append({"state": state, "start": date, "end": "null", "type": holiday})
    df_holidays = pl.read_csv(DATA_DIR / "ferien.csv", separator=";").with_columns(pl.col("start").str.to_date())
    pl.concat([df_holidays.with_columns(pl.col("end").str.to_date(format="%d.%m.%Y", strict=False))
                  , pl.DataFrame(holiday_list).with_columns(pl.col("end").str.to_date(format="%d.%m.%Y", strict=False))]
              ).write_csv(FEATURES_DIR / "holidays.csv")
