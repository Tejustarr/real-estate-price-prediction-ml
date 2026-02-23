import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.config import RAW_DATA_DIR


def load_data(save_raw=True):
    """
    Load California Housing dataset.
    Optionally save raw data to disk.
    """

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    if save_raw:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_csv(
            os.path.join(RAW_DATA_DIR, "california_housing.csv"),
            index=False
        )

    return df