"""
ChurnGuard — Create Feast Parquet Data
Converts the raw CSV into a parquet file that Feast can read as an offline store.

Run with: uv run python feature_repo/create_feast_data.py
"""

from datetime import UTC, datetime, timezone

import pandas as pd

# ── LOAD RAW DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/telco_churn.csv")
print(f"Loaded {len(df)} rows")

# ── SELECT ONLY THE COLUMNS FEAST NEEDS ──────────────────────────────────────
# Entity key + feature columns + target (we keep Churn for reference)
feature_cols = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "TechSupport",
    "PaymentMethod",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Churn",
]
df = df[feature_cols].copy()

# ── FIX DATA TYPES ────────────────────────────────────────────────────────────
# TotalCharges has 11 blank rows — convert to float, fill blanks with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

# ── ADD EVENT TIMESTAMP ───────────────────────────────────────────────────────
# Feast requires a timestamp column to know when features were recorded.
# Since this is historical/static data, we use a fixed timestamp.
# In a real system this would be the actual event time from your data pipeline.
df["event_timestamp"] = datetime(2024, 1, 1, tzinfo=UTC)

# ── SAVE AS PARQUET ───────────────────────────────────────────────────────────
# Parquet is Feast's preferred format for the offline store — faster than CSV,
# typed, and compressed.
output_path = "feature_repo/data/customer_features.parquet"

df["created"] = datetime(2024, 1, 1, tzinfo=timezone.utc)
df.to_parquet(output_path, index=False)

print(f"Saved {len(df)} rows to {output_path}")
print(f"Columns: {list(df.columns)}")
print("\nSample:")
print(df.head(3))