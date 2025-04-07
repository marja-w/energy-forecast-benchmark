# %% md
# # Features
# 
# This notebook creates features for the raw data.
# %% md
# ## Weather Features
# 
# Add features like temperature, humidity, sun hours, ...
# %% md
# Start with the daily data
# %%
import os

import polars as pl
import pgeocode
from meteostat import Point, Daily, Hourly

import holidays

from src.energy_forecast.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR, FEATURES_DIR, META_DIR

# %%

# INPUT PATHS
dataset_daily_csv_ = PROCESSED_DATA_DIR / "dataset_daily.csv"

# OUTPUT PATHS
city_df_path = FEATURES_DIR / "city_info.csv"
output_path_weather_daily = FEATURES_DIR / "weather_daily.csv"
output_path_weather_hourly = FEATURES_DIR / "weather_hourly.csv"

# PARAMETERS
holiday = True  # whether to create holiday features

# %%
data_df = pl.read_csv(dataset_daily_csv_).with_columns(pl.col("datetime").str.to_datetime())
# %% md
# Find time intervals for every city
# %%
data_df = data_df.with_columns(
    pl.coalesce(data_df.join(pl.read_csv(META_DIR / "kinergy_meta.csv"), on="id", how="left")["plz"],
                data_df.join(pl.read_csv(META_DIR / "legacy_meta.csv"), on="id", how="left")["plz"],
                data_df.join(pl.read_csv(META_DIR / "dh_meta.csv").rename({"eco_u_id": "id", "postal_code": "plz"}),
                             on="id", how="left")["plz"],
                ).str.strip_chars())
# %%
city_df = data_df.group_by(pl.col("plz")).agg(pl.col("datetime").min().alias("min_date"),
                                              pl.col("datetime").max().alias("max_date")).filter(
    ~(pl.col("plz") == "2700"))  # wien
city_df
# %% md
# Add coordinates to every city
# %%
if not os.path.exists(city_df_path):
    rows = list()
    for plz in city_df["plz"].unique():
        data = pgeocode.Nominatim("de").query_postal_code(str(plz))
        rows.append({"plz": plz, "lat": data["latitude"], "lon": data["longitude"], "state": data["state_code"]})

    info_df = pl.DataFrame(rows)
    city_df = city_df.join(info_df, on="plz", how="left")
    city_df.write_csv(city_df_path)
else:
    city_df = pl.read_csv(city_df_path, schema_overrides={"plz": pl.String}).with_columns(
        pl.col("min_date").str.to_datetime(),
        pl.col("max_date").str.to_datetime())
# %%
# %%
weather_dfs = list()
for row in city_df.iter_rows():
    start = row[1]
    end = row[2]
    loc = Point(row[3], row[4])

    data = Daily(loc, start, end)
    data = data.fetch()
    weather_dfs.append(pl.from_pandas(data.reset_index()).with_columns(pl.lit(row[0]).alias("plz")))
weather_df = pl.concat(weather_dfs)
weather_df

# %% md
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
# %% md
# Humidity is missing from Daily-data, we can retrieve hourly data and merge to daily data
# %%

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
# %% md
# Add to other weather data
# %%
weather_df = weather_df_hourly.join(weather_df, on=["plz", 'time'], how="left")
# %%
weather_df.write_csv(output_path_weather_daily)
# %% md
# Get hourly weather data as well
# %%

weather_dfs = list()
for row in city_df.iter_rows():
    start = row[1]
    end = row[2]
    loc = Point(row[3], row[4])

    data = Hourly(loc, start, end)
    data = data.fetch()
    weather_dfs.append(pl.from_pandas(data.reset_index()).with_columns(pl.lit(row[0]).alias("plz")))
weather_df_hourly = pl.concat(weather_dfs)
# %%
weather_df_hourly.write_csv(output_path_weather_hourly)
# %% md
# ## Time Features
# 
# School/University Break, Holidays
# %%
if holiday:
    holiday_dict = dict()
    ger_holidays = holidays.country_holidays("DE", years=range(2018, 2024))
    holiday_dict.update(ger_holidays)
    # %%
    holidays_state_dict = dict()
    for state in city_df["state"].unique():
        state_holidays = holidays.country_holidays("DE", subdiv=state, years=range(2018, 2024))
        holidays_state_dict.update({state: state_holidays})
    # %%

    holiday_list = list()
    for state in city_df["state"].unique():
        for date, holiday in holidays_state_dict[state].items():
            holiday_list.append({"state": state, "start": date, "end": "null", "type": holiday})
    # %%

    df_holidays = pl.read_csv(DATA_DIR / "ferien.csv", separator=";").with_columns(pl.col("start").str.to_date(),
                                                                                   pl.col("end").str.to_date())
    pl.concat([df_holidays, pl.DataFrame(holiday_list).cast({"end": pl.Date}, strict=False)]).write_csv(
        RAW_DATA_DIR / "holidays.csv")
    # %%
    data_df.with_columns(pl.col("date").dt.date().alias("date"))
    # %%
    data_df = data_df.with_columns(pl.col("date").dt.date().alias("date")
                                   ).with_columns(
        pl.when(pl.col("date").is_in(set(holiday_dict.keys()))).then(1).otherwise(0).alias("holiday"))
