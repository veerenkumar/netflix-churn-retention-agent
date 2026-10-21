"""Data preprocessing and scikit-learn transformation pipeline.
"""
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ID_COLUMN,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.feature_engineering import engineer_features


class ChurnDataPreprocessor:
    """Preprocesses subscription data, engineers behavioral features, and fits leakage-free

    scikit-learn transformation pipelines on training data.
    """

    def __init__(self):
        # Numerical features combining raw and engineered indicators
        self.numerical_cols: List[str] = RAW_NUMERICAL_FEATURES + [
            "watch_time_decline_pct",
            "marketing_engagement_rate",
            "viewing_intensity",
            "inactivity_indicator",
            "payment_risk_indicator",
            "complaint_risk_indicator",
        ]

        # Categorical features combining raw and engineered indicators
        self.categorical_cols: List[str] = RAW_CATEGORICAL_FEATURES + [
            "engagement_trend",
            "subscriber_tenure_group",
        ]

        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names_out: Optional[List[str]] = None
        self.is_fitted: bool = False

    def build_transformer(self) -> ColumnTransformer:
        """Constructs an imputation and encoding ColumnTransformer."""
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols),
            ],
            remainder="drop",
        )
        return transformer

    def prepare_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers features and cleans duplicates while preserving source columns."""
        df_clean = df.copy()
        if ID_COLUMN in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[ID_COLUMN], keep="first")
        df_engineered = engineer_features(df_clean)
        return df_engineered

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits raw dataset into stratified training and testing sets.

        Args:
            df: Input raw or engineered dataframe with target column.
            test_size: Proportion of test split (default: 0.20).
            random_state: Reproducible seed.

        Returns:
            Tuple: (df_train, df_test, y_train, y_test)
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' required for train/test split.")

        df_prepared = self.prepare_raw_dataframe(df)
        y = df_prepared[TARGET_COLUMN].astype(int)

        df_train, df_test, y_train, y_test = train_test_split(
            df_prepared,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return df_train, df_test, y_train, y_test

    def fit(self, df_train: pd.DataFrame) -> "ChurnDataPreprocessor":
        """Fits preprocessing pipelines strictly on training data."""
        df_prepared = self.prepare_raw_dataframe(df_train)
        self.transformer = self.build_transformer()
        self.transformer.fit(df_prepared)

        # Retrieve resulting feature names after one-hot encoding
        try:
            self.feature_names_out = list(self.transformer.get_feature_names_out())
        except Exception:
            # Fallback if get_feature_names_out fails
            self.feature_names_out = [f"feat_{i}" for i in range(len(self.numerical_cols))]

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms subscriber data into scaled/encoded matrix for model consumption."""
        if not self.is_fitted or self.transformer is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df_prepared = self.prepare_raw_dataframe(df)
        X_matrix = self.transformer.transform(df_prepared)
        return X_matrix

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fits preprocessor on training data and transforms it in one step."""
        self.fit(df_train)
        return self.transform(df_train)


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data

    sample_data = generate_synthetic_churn_data(500, random_seed=12)
    preproc = ChurnDataPreprocessor()
    train_df, test_df, y_tr, y_te = preproc.split_data(sample_data, test_size=0.2)
    X_tr = preproc.fit_transform(train_df)
    X_te = preproc.transform(test_df)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Total encoded features: {len(preproc.feature_names_out)}")
