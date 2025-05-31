from src.energy_forecast.data_processing.feature_generation import generate_weather_dfs, generate_holiday_df
from src.energy_forecast.data_processing.lod_data_processor import generate_lod_df


def generate_weather_features(res: str = "hourly", interpolated: str = ""):
    """Generate weather dfs and save them as .csv files"""
    generate_weather_dfs(res, interpolated)

def generate_holiday_feature():
    """Generate holidays.csv"""
    generate_holiday_df()  # TODO: missing data/ferien.csv

def generate_lod_data():
    """Generate lod data for KI in Fernwärme project data"""
    generate_lod_df()

if __name__ == '__main__':
    res = "hourly"
    interpolated = ""
    generate_weather_features(res=res, interpolated=interpolated)
    # generate_holiday_feature()
    # generate_lod_data()