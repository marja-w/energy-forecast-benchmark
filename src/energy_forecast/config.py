from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]  # energy-forecast-wahl/src/energy_forecast
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

CATEGORICAL_FEATURES = ["typ"]
CATEGORICAL_FEATURES_BINARY = ["weekend", "holiday"]
CONTINUOUS_FEATURES = ["diff_t-1", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
                       'wspd', 'wpgt', 'pres', 'tsun', "daily_avg", "heated_area", "anzahlwhg", "ground_surface",
                       "building_height", "storeys_above_ground"]

## Feature Configs
FEATURE_SET_1 = ["diff", "diff_t-1"]
FEATURE_SET_2 = ["diff", "diff_t-1", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
                 'wspd', 'wpgt', 'pres', 'tsun']
FEATURE_SET_3 = ["diff", 'hum_avg', 'hum_min', 'hum_max', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd',
                 'wpgt', 'pres', 'tsun']
FEATURE_SET_4 = ["diff", "diff_t-1", "tmax", "tsun", "wpgt", "hum_avg"]
FEATURE_SET_5 = ["diff", "tmax", "tsun", "wpgt", "hum_avg"]
FEATURE_SET_6 = ["diff", "diff_t-1", "tmax", "tsun", "wpgt", "hum_avg", "daily_avg", "heated_area", "anzahlwhg", "typ"]
FEATURE_SETS = {1: FEATURE_SET_1, 2: FEATURE_SET_2, 3: FEATURE_SET_3, 4: FEATURE_SET_4, 5: FEATURE_SET_5,
                6: FEATURE_SET_6}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
