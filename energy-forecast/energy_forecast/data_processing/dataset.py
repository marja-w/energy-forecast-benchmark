from pathlib import Path

import pandas as pd
import polars as pl
import typer
from loguru import logger

from src.energy_forecast.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.energy_forecast.util import get_season

app = typer.Typer()


def load_counter_data(input_path_counter: Path, store: bool = False):
    """

    :param input_path_counter: path to file with counter data
    :param store: set to true if you want to store the DataFrame as .csv
    :return: dataframe with counter data
    """
    data = pd.read_csv(input_path_counter, low_memory=False)

    try:
        data = data.drop(columns=['lastAggregated'])
    except KeyError:
        pass
    data['ArrivalTime'] = pd.to_datetime(data['ArrivalTime'])
    try:
        data.set_index(["GSM_ID", "TP", "Tag", "ArrivalTime"], inplace=True)
        data = data[~data.index.duplicated(keep="first")]
        data['diff'] = data.groupby(['GSM_ID', 'TP', 'Tag'])[
            'Val'].diff().dropna()  # compute difference TODO: Datenlücken bei date? check
    except KeyError:
        # kinergy data does not have GSM_ID, TP, or Tag column
        # diff already computed
        data.set_index(["hash", "ArrivalTime"], inplace=True)
        data = data[~data.index.duplicated(keep="first")]


    data['dayofweek'] = data.index.get_level_values('ArrivalTime').dayofweek
    data['weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    data['season'] = data.apply(lambda row: get_season(row['GMonth'], row['GDay']), axis=1)

    data = data.reset_index()

    if store: data.to_csv(PROCESSED_DATA_DIR / "consumption-cleaned-kinergy.csv")
    return data


def load_env_data(input_path_env: Path):
    """

    :param input_path_env: path to file with environment data
    :return: dataframe with environment data
    """
    wetter_df = pd.read_csv(input_path_env)
    wetter_df.rename(columns={"City": "ort"}, inplace=True)
    return wetter_df


def merge_datasets(counter_df: pd.DataFrame, env_df: pd.DataFrame, store: bool = False):
    """

    :param counter_df:
    :param env_df:
    :param store: set to true if you want to store the DataFrame as .csv
    :return: dataframe with merged data
    """
    df = counter_df.merge(env_df, on=["ort", "GYear", "GMonth", "GDay"], how="left")
    df['Humidity_min'] = df['Humidity_min'].str.replace("%", "")  # replace the '%' sign added in some rows
    df['Humidity_max'] = df['Humidity_max'].str.replace("%", "")  # replace the '%' sign added in some rows
    try:
        df['plz'] = df['plz'].str.strip()  # replace empty chars before and after
    except AttributeError:
        pass
    if store: df.round(1).to_csv(PROCESSED_DATA_DIR / "consumption-cleaned-merged-weather-kinergy.csv")
    return df


@app.command()
def main(
        input_path_counter: Path = RAW_DATA_DIR / "kinergy_daily_values.csv",
        input_path_env: Path = RAW_DATA_DIR / "legacy_system_environment_daily_values.csv",
        output_path: Path = PROCESSED_DATA_DIR / "dataset-kinergy.csv"
):
    logger.info("Loading data")
    counter_df = load_counter_data(input_path_counter, store=True)
    env_df = load_env_data(input_path_env)
    df_pandas: pd.DataFrame = merge_datasets(counter_df, env_df, store=True)

    print(f"Current length of data: {len(df_pandas)}")

    df: pl.DataFrame = pl.from_pandas(df_pandas)  # transform to polars dataframe
    df.write_csv(output_path)


if __name__ == "__main__":
    app()
