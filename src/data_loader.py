"""Loading the raw dataset."""
from pathlib import Path
import pandas as pd
from src.config import DATA_PATH, TARGET, FEATURES


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as supplied.

    IMPORTANT: pandas normally turns the text "None" into NaN. In this dataset
    "None" is a REAL category of alcohol_consumption (no alcohol), so we switch
    that default off and only treat truly empty cells as missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}. Put it in the data/ folder.")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    return df
