from datetime import timedelta

from src.energy_forecast.config import RAW_DATA_DIR
import polars as pl

from src.energy_forecast.util import get_missing_dates, find_time_spans


def remove_error_values(data_df: pl.DataFrame) -> pl.DataFrame:
    missing_dates = get_missing_dates(data_df)
    for row in missing_dates.iter_rows():
        id = row[0]
        missing_date_sens = row[1]
        time_spans = find_time_spans(missing_date_sens)
        for time_span in time_spans.iter_rows():
            end_date = time_span[1]
            corrupt_date = end_date + timedelta(days=1)  # the corrupt date should be the one the day after

if __name__ == '__main__':
    dh_data_file = RAW_DATA_DIR / "district_heating_daily.csv"
    df = pl.read_csv(dh_data_file)

    # remove corrupt sensors
    corr_sensors = ["Kielort 14", "Moorbekstraße 17", "Kielort 21", "Kielortring 14", "Kielortring 22",
                    "Kielortring 16",
                    "Ulzburger Straße 459 A", "Ulzburger Straße 461", "Kielort 22", "Kielort 19", "Kielortring 51",
                    "Kielort 16", "Friedrichsgaber Weg 453"]
    df = df.filter(~pl.col("adresse").is_in(corr_sensors))

    # remove errors resulting from connection errors
    df = remove_error_values(df)

    # remove flatlines
