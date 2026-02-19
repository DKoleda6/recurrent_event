"""Small IO helpers for loading saved imputed datasets."""
from typing import List
import pandas as pd


def load_imputations_from_csv(paths: List[str]):
    """Return list of DataFrames for each imputed CSV path."""
    return [pd.read_csv(p) for p in paths]