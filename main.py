"""Implementation of the Recency-weighted historical forecast solution to the Run-way Functions:
Predict Reconfigurations at US Airports challenge.

https://www.drivendata.co/blog/airport-configuration-benchmark/
"""

import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import typer
from src.utils import make_all_predictions, get_needed_data


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
feature_directory = Path("./data")
prediction_path = Path("./prediction.csv")
model_assets_dir = Path("./src/model_assets")


def main(prediction_time: datetime):
    logger.info("Computing my predictions for {}", prediction_time)

    airport_directories = sorted(path for path in feature_directory.glob("k*"))

    submission_format = pd.read_csv(
        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    )
    print(f"{len(submission_format):,} rows x {len(submission_format.columns)} columns")

    submission = submission_format.copy().reset_index(drop=True)
    submission["active"] = np.nan
    submission = submission.astype({'config': 'str'})

    airport_config_df_map, airport_weather_df_map, unique_config_maps, airport_codes = get_needed_data(airport_directories, submission)

    make_all_predictions(airport_config_df_map, airport_weather_df_map, model_assets_dir, submission)

    submission = submission.fillna(0)
    submission.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)


if __name__ == "__main__":
    typer.run(main)
