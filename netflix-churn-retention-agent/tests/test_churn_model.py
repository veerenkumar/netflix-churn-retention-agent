"""Automated unit tests for machine learning churn model training, inference, and persistence.
"""
from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.churn_model import ChurnModelPipeline
from src.data_generator import generate_synthetic_churn_data


def test_model_training_and_evaluation():
    """Verifies that model training compares both candidates and computes metrics."""
    raw = generate_synthetic_churn_data(500, random_seed=42)
    pipeline = ChurnModelPipeline()
    meta = pipeline.train_and_evaluate(raw, test_size=0.20)

    assert pipeline.is_trained
    assert pipeline.best_model_name in ["Logistic Regression", "Random Forest"]
    assert "evaluation_results" in meta
    assert "Logistic Regression" in meta["evaluation_results"]
    assert "Random Forest" in meta["evaluation_results"]

    for m_name, res in meta["evaluation_results"].items():
        assert 0.0 <= res["accuracy"] <= 1.0
        assert 0.0 <= res["f1_score"] <= 1.0
        assert 0.0 <= res["roc_auc"] <= 1.0


def test_predict_proba_bounds():
    """Verifies that estimated probabilities are strictly within [0.0, 1.0]."""
    raw = generate_synthetic_churn_data(400, random_seed=42)
    pipeline = ChurnModelPipeline()
    pipeline.train_and_evaluate(raw)

    probs = pipeline.predict_proba(raw)
    assert len(probs) == len(raw)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_untrained_model_raises_error():
    """Ensures untrained model raises RuntimeError when predict is attempted."""
    raw = generate_synthetic_churn_data(100, random_seed=42)
    pipeline = ChurnModelPipeline()
    with pytest.raises(RuntimeError):
        pipeline.predict_proba(raw)


def test_model_saving_and_loading():
    """Tests saving model artifact to disk and reloading it for inference."""
    raw = generate_synthetic_churn_data(400, random_seed=42)
    pipeline = ChurnModelPipeline()
    pipeline.train_and_evaluate(raw)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_file = Path(tmp_dir) / "test_model.joblib"
        tmp_meta_file = Path(tmp_dir) / "test_meta.json"

        pipeline.save(tmp_model_file, tmp_meta_file)
        assert tmp_model_file.exists()
        assert tmp_meta_file.exists()

        loaded_pipeline = ChurnModelPipeline.load(tmp_model_file)
        assert loaded_pipeline.is_trained
        assert loaded_pipeline.best_model_name == pipeline.best_model_name

        p1 = pipeline.predict_proba(raw.head(10))
        p2 = loaded_pipeline.predict_proba(raw.head(10))
        np.testing.assert_allclose(p1, p2, rtol=1e-5)
