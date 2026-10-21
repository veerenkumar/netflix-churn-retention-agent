"""Data loading, schema inspection, and defensive validation module.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union
import io
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_DATASET_PATH, ID_COLUMN, REQUIRED_COLUMNS, TARGET_COLUMN


def load_dataset(source: Union[str, Path, io.BytesIO, io.StringIO, pd.DataFrame]) -> pd.DataFrame:
    """Loads a subscriber dataset from a file path, uploaded file buffer, or existing DataFrame.

    Args:
        source: File path, buffer, or DataFrame.

    Returns:
        pd.DataFrame: Loaded customer dataset.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")
        return pd.read_csv(path)
    return pd.read_csv(source)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Defensively validates subscriber dataset for structure, types, missing values, and integrity.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple:
            - is_valid (bool): True if schema and core constraints are met.
            - errors (List[str]): Critical issues preventing model training or scoring.
            - warnings (List[str]): Non-critical issues or data hygiene notices.
            - summary (Dict[str, Any]): Data health summary metrics.
    """
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {
        "total_rows": 0,
        "total_columns": 0,
        "duplicate_ids": 0,
        "missing_values_total": 0,
        "observed_churn_rate": None,
    }

    if df is None or df.empty:
        errors.append("Dataset is empty or null.")
        return False, errors, warnings, summary

    summary["total_rows"] = len(df)
    summary["total_columns"] = len(df.columns)

    # 1. Required Columns Check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # 2. Check Customer ID Uniqueness
    if ID_COLUMN in df.columns:
        duplicate_count = df[ID_COLUMN].duplicated().sum()
        summary["duplicate_ids"] = int(duplicate_count)
        if duplicate_count > 0:
            warnings.append(
                f"Found {duplicate_count} duplicate '{ID_COLUMN}' entries. These should be deduplicated."
            )

    # 3. Missing Value Audit
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    summary["missing_values_total"] = total_nulls
    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        warnings.append(f"Missing values detected across columns: {cols_with_nulls}")

    # 4. Target Label Validation (if present)
    if TARGET_COLUMN in df.columns:
        unique_targets = df[TARGET_COLUMN].dropna().unique()
        if not set(unique_targets).issubset({0, 1}):
            errors.append(
                f"Target column '{TARGET_COLUMN}' must contain only binary values (0 or 1). Found: {list(unique_targets)}"
            )
        else:
            summary["observed_churn_rate"] = float(df[TARGET_COLUMN].mean())

    # 5. Sanity Bounds on Numerical Metrics
    if "monthly_subscription_fee" in df.columns:
        if (df["monthly_subscription_fee"] < 0).any():
            errors.append("Monthly subscription fee cannot be negative.")

    if "active_days_last_30_days" in df.columns:
        if (df["active_days_last_30_days"] < 0).any() or (df["active_days_last_30_days"] > 31).any():
            errors.append("Active days in last 30 days must be within [0, 31].")

    if "watch_hours_last_30_days" in df.columns:
        if (df["watch_hours_last_30_days"] < 0).any():
            errors.append("Watch hours cannot be negative.")

    if "days_since_last_activity" in df.columns:
        if (df["days_since_last_activity"] < 0).any():
            errors.append("Days since last activity cannot be negative.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, summary


if __name__ == "__main__":
    from config import DEFAULT_DATASET_PATH

    print("Validating default dataset...")
    df_sample = load_dataset(DEFAULT_DATASET_PATH)
    valid, errs, warns, stats = validate_dataset(df_sample)
    print(f"Valid: {valid}")
    print(f"Errors: {errs}")
    print(f"Warnings: {warns}")
    print(f"Summary: {stats}")
