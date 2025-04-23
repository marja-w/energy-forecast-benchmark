from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# other variables
N_CLUSTER = 3  # number of clusters for clustering
PADDING_VALUE = -99  # value for padding target variables of a time series when creating training data

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]  # energy-forecast-wahl/src/energy_forecast
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

FEATURES_DIR = EXTERNAL_DATA_DIR / "features"
META_DIR = INTERIM_DATA_DIR / "meta"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

CATEGORICAL_FEATURES = ["typ", "primary_energy"]
CATEGORICAL_FEATURES_BINARY = ["weekend", "holiday"]
CONTINUOUS_FEATURES = ["diff_t-1", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
                       'wspd', 'wpgt', 'pres', 'tsun', "daily_avg", "heated_area", "anzahlwhg", "ground_surface",
                       "building_height", "storeys_above_ground"]
CONTINUOUS_FEATURES_CYCLIC = ["weekday", "day_of_month"]
FEATURES = ["diff"] + CATEGORICAL_FEATURES + CONTINUOUS_FEATURES + CATEGORICAL_FEATURES_BINARY + CONTINUOUS_FEATURES_CYCLIC

## Feature Configs
FEATURES_DIFF = ["diff", "diff_t-1"]
FEATURES_WEATHER = ['hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd',
                    'wpgt', 'pres', 'tsun']
FEATURES_WEATHER_NO_SNOW = list(set(FEATURES_WEATHER) - {"snow"})
FEATURES_WEATHER = FEATURES_WEATHER_NO_SNOW  # snow has lots of missing values
FEATURES_WEATHER_SIGNIFICANT = ["tmax", "tsun", "wpgt", "hum_avg"]
FEATURES_BUILDING = ["daily_avg", "heated_area", "anzahlwhg", "typ", "primary_energy"]
FEATURES_BUILDING_NO_APPARTMENT = list(set(FEATURES_BUILDING) - {"anzahlwhg"})
FEATURES_TIME = CATEGORICAL_FEATURES_BINARY + ["weekday", "day_of_month"]
FEATURES_DH = ["ground_surface", "building_height", "storeys_above_ground", "heated_area"]
FEATURES_ALL = FEATURES_DIFF + FEATURES_WEATHER + FEATURES_BUILDING + FEATURES_TIME

FEATURE_SET_1 = FEATURES_DIFF
FEATURE_SET_2 = FEATURES_DIFF + FEATURES_WEATHER
FEATURE_SET_3 = ["diff"] + FEATURES_WEATHER
FEATURE_SET_4 = FEATURES_DIFF + FEATURES_WEATHER_SIGNIFICANT
FEATURE_SET_5 = ["diff"] + FEATURES_WEATHER_SIGNIFICANT
FEATURE_SET_6 = FEATURES_DIFF + FEATURES_WEATHER_SIGNIFICANT + FEATURES_BUILDING
FEATURE_SET_7 = FEATURES_ALL
FEATURE_SET_8 = FEATURES_DIFF + FEATURES_WEATHER + FEATURES_TIME + ["daily_avg"]
FEATURE_SET_9 = FEATURES_DIFF + FEATURES_WEATHER + FEATURES_BUILDING + ["weekend"]
FEATURE_SET_10 = FEATURES_DIFF + FEATURES_WEATHER + FEATURES_TIME + FEATURES_BUILDING_NO_APPARTMENT
FEATURE_SET_11 = FEATURES_DIFF + FEATURES_WEATHER + FEATURES_TIME + FEATURES_DH
FEATURE_SET_12 = ["diff"]

FEATURE_SETS = {1: FEATURE_SET_1, 2: FEATURE_SET_2, 3: FEATURE_SET_3, 4: FEATURE_SET_4, 5: FEATURE_SET_5,
                6: FEATURE_SET_6, 7: FEATURE_SET_7, 8: FEATURE_SET_8, 9: FEATURE_SET_9, 10: FEATURE_SET_10,
                11: FEATURE_SET_11, 12: FEATURE_SET_12}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
