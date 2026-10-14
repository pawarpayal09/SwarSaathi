"""Cleaning and auditing: invalid values, duplicates, leakage checks."""
import pandas as pd
from src.config import (FEATURES, TARGET, VALID_RANGES, NUMERIC_FEATURES,
                        CATEGORICAL_FEATURES)


def audit_data(df: pd.DataFrame) -> dict:
    """Collect verified facts about the raw data (used in reports and README)."""
    unique_df = df.drop_duplicates()
    # Feature combinations that appear with more than one target label
    conflicts = df.groupby(FEATURES, dropna=False)[TARGET].nunique()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "unique_rows": int(len(unique_df)),
        "rows_duplicated_on_features_only": int(df.duplicated(FEATURES).sum()),
        "feature_combinations_with_conflicting_targets": int((conflicts > 1).sum()),
        "missing_values_per_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "target_distribution_original": df[TARGET].value_counts().to_dict(),
        "target_distribution_after_dedup": unique_df[TARGET].value_counts().to_dict(),
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
        "observed_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_FEATURES},
    }


def strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stray spaces from category text so 'High ' and 'High' match."""
    cleaned = df.copy()
    for col in CATEGORICAL_FEATURES + [TARGET]:
        cleaned[col] = cleaned[col].astype("string").str.strip().astype(object)
    return cleaned


def mark_invalid_values(df: pd.DataFrame):
    """Set impossible numeric values (see config.VALID_RANGES) to NaN.

    Returns (cleaned_df, log). The target column is never touched.
    """
    cleaned = df.copy()
    log = {}
    for col, (low, high) in VALID_RANGES.items():
        bad = (cleaned[col] < low) | (cleaned[col] > high)
        if bad.any():
            log[col] = {"count": int(bad.sum()), "values": cleaned.loc[bad, col].tolist()[:10]}
            cleaned.loc[bad, col] = float("nan")
    return cleaned, log


def remove_exact_duplicates(df: pd.DataFrame):
    """Drop exact duplicate rows (done BEFORE the train/test split).

    Why: the original file repeats the same record many times. If copies land
    in both train and test, the model is tested on rows it already saw, which
    inflates scores (data leakage).
    """
    before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, {"rows_before": before, "rows_after": len(deduped),
                     "rows_removed": before - len(deduped)}


def prepare_dataset(df: pd.DataFrame):
    """Cleaning order: strip text -> drop exact duplicates -> flag invalid values.

    Duplicates are removed on the values AS SUPPLIED, so a row is only dropped
    if it is truly identical to another. Invalid values are blanked afterwards
    (imputation itself happens later, inside the pipeline, on training data only).
    """
    cleaned = strip_text_columns(df)
    deduped, dup_log = remove_exact_duplicates(cleaned)
    deduped, invalid_log = mark_invalid_values(deduped)
    return deduped, {"deduplication": dup_log, "invalid_values": invalid_log}


def check_split_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    """Count test rows whose features are identical to a training row (should be 0)."""
    cols = list(X_train.columns)
    merged = X_test.fillna(-999).merge(X_train.fillna(-999).drop_duplicates(), on=cols, how="inner")
    return int(len(merged))
