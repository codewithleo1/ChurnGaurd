"""
ChurnGuard — Test Suite
Tests for data, preprocessing, training, and feature store.
Run with: uv run pytest tests/ -v
"""

import pandas as pd
import pytest
from pathlib import Path


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Load the raw dataset once for all tests that need it."""
    path = Path("data/raw/telco_churn.csv")
    assert path.exists(), "Raw dataset not found — run DVC pull"
    return pd.read_csv(path)


@pytest.fixture
def preprocessed(raw_df):
    """Return preprocessed X, y using the actual training script logic."""
    import sys
    sys.path.insert(0, "training")
    from train import preprocess
    return preprocess(raw_df)


# ── DATA TESTS ────────────────────────────────────────────────────────────────

class TestRawData:
    def test_row_count(self, raw_df):
        """Dataset should have exactly 7043 rows."""
        assert len(raw_df) == 7043

    def test_required_columns(self, raw_df):
        """All expected columns must be present."""
        required = [
            "customerID", "tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "InternetService", "Churn"
        ]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_churn_column_values(self, raw_df):
        """Churn column should only contain Yes/No."""
        assert set(raw_df["Churn"].unique()) == {"Yes", "No"}

    def test_class_imbalance(self, raw_df):
        """Churn rate should be roughly 26-27%."""
        churn_rate = (raw_df["Churn"] == "Yes").mean()
        assert 0.25 < churn_rate < 0.28, f"Unexpected churn rate: {churn_rate:.2%}"

    def test_blank_total_charges(self, raw_df):
        """TotalCharges should have exactly 11 blank rows."""
        blanks = (raw_df["TotalCharges"] == " ").sum()
        assert blanks == 11, f"Expected 11 blank TotalCharges rows, got {blanks}"

    def test_tenure_non_negative(self, raw_df):
        """Tenure should never be negative."""
        assert (raw_df["tenure"] >= 0).all()

    def test_monthly_charges_positive(self, raw_df):
        """MonthlyCharges should always be positive."""
        assert (raw_df["MonthlyCharges"] > 0).all()


# ── PREPROCESSING TESTS ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shapes(self, preprocessed):
        """X should have 19 features, y should be a 1D series."""
        X, y = preprocessed
        assert X.shape[1] == 19
        assert len(y.shape) == 1

    def test_no_nulls_after_preprocessing(self, preprocessed):
        """No nulls should remain after preprocessing."""
        X, y = preprocessed
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_target_is_binary(self, preprocessed):
        """Target should only contain 0 and 1."""
        _, y = preprocessed
        assert set(y.unique()) == {0, 1}

    def test_no_string_columns(self, preprocessed):
        """All columns should be numeric after encoding."""
        X, _ = preprocessed
        string_cols = X.select_dtypes(include="object").columns.tolist()
        assert len(string_cols) == 0, f"String columns remain: {string_cols}"

    def test_customer_id_dropped(self, preprocessed):
        """customerID should be dropped — it's not a feature."""
        X, _ = preprocessed
        assert "customerID" not in X.columns


# ── TRAINING TESTS ────────────────────────────────────────────────────────────

class TestTraining:
    def test_model_metrics(self, preprocessed):
        """Trained model should meet minimum quality thresholds."""
        import sys
        sys.path.insert(0, "training")
        from train import train

        X, y = preprocessed
        _, metrics = train(X, y)

        # Minimum bars — if these fail, something is wrong with the pipeline
        assert metrics["roc_auc"] > 0.75, f"ROC-AUC too low: {metrics['roc_auc']:.3f}"
        assert metrics["recall"] > 0.65, f"Recall too low: {metrics['recall']:.3f}"
        assert metrics["accuracy"] > 0.70, f"Accuracy too low: {metrics['accuracy']:.3f}"

    def test_model_predicts_both_classes(self, preprocessed):
        """Model should predict both churn and no-churn — not just one class."""
        import sys
        sys.path.insert(0, "training")
        from train import train
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X, y = preprocessed
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(set(preds)) == 2, "Model only predicts one class"


# ── FEATURE STORE TESTS ───────────────────────────────────────────────────────

class TestFeatureStore:
    def test_parquet_exists(self):
        """Feast parquet file should exist."""
        path = Path("feature_repo/data/customer_features.parquet")
        assert path.exists(), "Run create_feast_data.py first"

    def test_parquet_has_required_columns(self):
        """Parquet should have all required Feast columns."""
        df = pd.read_parquet("feature_repo/data/customer_features.parquet")
        required = ["customerID", "event_timestamp", "created", "tenure",
                    "MonthlyCharges", "Contract"]
        for col in required:
            assert col in df.columns, f"Missing column in parquet: {col}"

    def test_parquet_row_count(self):
        """Parquet should have same row count as raw CSV."""
        df = pd.read_parquet("feature_repo/data/customer_features.parquet")
        assert len(df) == 7043

    def test_online_store_lookup(self):
        """Online store should return features for a known customer."""
        from feast import FeatureStore
        store = FeatureStore(repo_path="feature_repo")

        result = store.get_online_features(
            features=["customer_features:tenure", "customer_features:Contract"],
            entity_rows=[{"customerID": "7590-VHVEG"}],
        ).to_dict()

        assert result["tenure"][0] == 1
        assert result["Contract"][0] == "Month-to-month"