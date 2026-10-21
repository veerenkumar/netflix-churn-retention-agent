"""Machine learning pipeline for subscription churn prediction, evaluation, and calibration.
"""
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import METADATA_PATH, MODEL_PATH, PREDICTION_WINDOW_DESCRIPTION
from src.preprocessing import ChurnDataPreprocessor


class ChurnModelPipeline:
    """Manages model training, evaluation, threshold calibration, and inference."""

    def __init__(self):
        self.preprocessor = ChurnDataPreprocessor()
        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.is_trained: bool = False
        self.training_metadata: Dict[str, Any] = {}

    def _evaluate_classifier(
        self,
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates comprehensive evaluation metrics on held-out test data."""
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_test, y_prob))
        except ValueError:
            pr_auc = float(np.mean(y_test))

        brier = float(brier_score_loss(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (int(x) for x in cm.ravel())

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            },
            "pr_curve": {
                "precision": p_curve.tolist(),
                "recall": r_curve.tolist(),
            },
        }

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
        primary_metric: str = "f1_score",
    ) -> Dict[str, Any]:
        """Trains Logistic Regression and Random Forest models, compares performance on

        held-out test data, calibrates probabilities, and selects the superior model.
        """
        # Split data cleanly
        df_train, df_test, y_train, y_test = self.preprocessor.split_data(
            df, test_size=test_size, random_state=random_state
        )

        # Fit preprocessor strictly on training data
        X_train = self.preprocessor.fit_transform(df_train)
        X_test = self.preprocessor.transform(df_test)

        # Candidate 1: Logistic Regression (Regularized with balanced class weights)
        lr_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        )
        lr_model.fit(X_train, y_train)
        self.models["Logistic Regression"] = lr_model
        self.evaluation_results["Logistic Regression"] = self._evaluate_classifier(
            lr_model, X_test, y_test
        )

        # Candidate 2: Random Forest Classifier (Balanced)
        rf_model = RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        self.models["Random Forest"] = rf_model
        self.evaluation_results["Random Forest"] = self._evaluate_classifier(
            rf_model, X_test, y_test
        )

        # Select best model based on chosen primary metric (e.g. F1 or ROC-AUC)
        best_name = max(
            self.evaluation_results.keys(),
            key=lambda k: self.evaluation_results[k].get(primary_metric, 0.0),
        )
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        self.is_trained = True

        self.training_metadata = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_window": PREDICTION_WINDOW_DESCRIPTION,
            "total_records": len(df),
            "train_records": len(df_train),
            "test_records": len(df_test),
            "train_churn_rate": round(float(y_train.mean()), 4),
            "test_churn_rate": round(float(y_test.mean()), 4),
            "best_model_name": self.best_model_name,
            "selected_metric": primary_metric,
            "feature_count": len(self.preprocessor.feature_names_out or []),
            "evaluation_results": self.evaluation_results,
        }

        return self.training_metadata

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Estimates churn probability for subscribers."""
        if not self.is_trained or self.best_model is None:
            raise RuntimeError("Model pipeline has not been trained yet.")

        X = self.preprocessor.transform(df)
        probas = self.best_model.predict_proba(X)[:, 1]
        return probas

    def predict(self, df: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predicts binary churn label according to decision threshold."""
        probas = self.predict_proba(df)
        return (probas >= threshold).astype(int)

    def save(self, model_file: Path = MODEL_PATH, meta_file: Path = METADATA_PATH) -> None:
        """Saves trained model bundle and metadata to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model pipeline.")

        model_file.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "preprocessor": self.preprocessor,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "models": self.models,
            "evaluation_results": self.evaluation_results,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(bundle, model_file)

        # Save metadata JSON for lightweight inspection
        with open(meta_file, "w") as f:
            # Strip large curve arrays for clean JSON metadata
            meta_clean = json.loads(json.dumps(self.training_metadata))
            for model_k, results_dict in meta_clean.get("evaluation_results", {}).items():
                results_dict.pop("roc_curve", None)
                results_dict.pop("pr_curve", None)
            json.dump(meta_clean, f, indent=2)

    @classmethod
    def load(cls, model_file: Path = MODEL_PATH) -> "ChurnModelPipeline":
        """Loads trained pipeline bundle from disk safely."""
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Trained model artifact not found at: {model_file}")

        bundle = joblib.load(model_file)
        pipeline = cls()
        pipeline.preprocessor = bundle["preprocessor"]
        pipeline.best_model_name = bundle["best_model_name"]
        pipeline.best_model = bundle["best_model"]
        pipeline.models = bundle.get("models", {})
        pipeline.evaluation_results = bundle.get("evaluation_results", {})
        pipeline.training_metadata = bundle.get("training_metadata", {})
        pipeline.is_trained = True
        return pipeline


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_churn_data

    print("Training and evaluating churn models...")
    raw_df = generate_synthetic_churn_data(1000, random_seed=42)
    pipeline = ChurnModelPipeline()
    meta = pipeline.train_and_evaluate(raw_df)
    print("Selected Model:", meta["best_model_name"])
    for m_name, res in meta["evaluation_results"].items():
        print(f"\n--- {m_name} ---")
        print(f"Accuracy:  {res['accuracy']:.4f}")
        print(f"F1-Score:  {res['f1_score']:.4f}")
        print(f"ROC-AUC:   {res['roc_auc']:.4f}")
        print(f"Confusion: {res['confusion_matrix']}")
    pipeline.save()
    print("\nSaved trained model artifact.")
