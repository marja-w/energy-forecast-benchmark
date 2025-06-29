import pandas as pd
import wandb

from src.energy_forecast.config import REPORTS_DIR

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("rausch-technology/ma-wahl-forecast")
id_list = ["l2zri11o", "a11okqcm", "sk1e5aar", "bdq9efyy"]

summary_list, config_list, name_list = [], [], []
for run in runs:
    if run.id not in id_list:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv(REPORTS_DIR / "best_models_7_day_forecast.csv")