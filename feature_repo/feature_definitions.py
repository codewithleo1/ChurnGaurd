"""
ChurnGuard — Feast Feature Definitions
Defines the entity, data source, and feature views for the churn prediction model.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float64, Int64, String, ValueType

# ── PROJECT ──────────────────────────────────────────────────────────────────
# Top-level Feast project — groups all feature definitions together
project = Project(name="churnguard", description="Customer churn prediction features")

# ── ENTITY ───────────────────────────────────────────────────────────────────
# The thing we're predicting about — a customer identified by customerID.
# Feast uses this as the primary key to look up features in both stores.

customer = Entity(
    name="customer",
    join_keys=["customerID"],
    value_type=ValueType.STRING,
    description="A telecom customer",
)

# ── DATA SOURCE ───────────────────────────────────────────────────────────────
# Where the offline feature data lives — a parquet file for local development.
# In production this would point to BigQuery, Redshift, Snowflake, etc.
# We'll generate this parquet file from our CSV in the next step.
customer_stats_source = FileSource(
    name="customer_stats_source",
    path="data/customer_features.parquet",   # relative to feature_repo/
    timestamp_field="event_timestamp",        # Feast requires a timestamp column
)

customer_stats_source = FileSource(
    name="customer_stats_source",
    path="data/customer_features.parquet",    # relative to feature_repo/
    timestamp_field="event_timestamp",        # Feast requires a timestamp column
    created_timestamp_column="created",
)

# ── FEATURE VIEW ──────────────────────────────────────────────────────────────
# A named group of features for a given entity.
# Same feature logic is used for both training (offline) and serving (online).
# This is what prevents training/serving skew.
customer_features_fv = FeatureView(
    name="customer_features",
    entities=[customer],
    # TTL: how long a feature value stays valid in the online store.
    # 90 days — if a customer hasn't been seen in 90 days, their features expire.
    ttl=timedelta(days=3650),  # 10 years — static dataset, no expiry concern
    schema=[
        # How long the customer has been with the company (months)
        Field(name="tenure", dtype=Int64),
        # Monthly bill amount — higher bills correlate with churn (from EDA)
        Field(name="MonthlyCharges", dtype=Float64),
        # Total charges — we keep it here even though it correlates with tenure
        Field(name="TotalCharges", dtype=Float64),
        # Contract type: Month-to-month customers churn at 42.7% (from EDA)
        Field(name="Contract", dtype=String),
        # Internet service type
        Field(name="InternetService", dtype=String),
        # Whether customer has tech support — affects satisfaction
        Field(name="TechSupport", dtype=String),
        # Payment method — electronic check customers churn more
        Field(name="PaymentMethod", dtype=String),
        # Whether customer is a senior citizen
        Field(name="SeniorCitizen", dtype=Int64),
        # Whether customer has a partner
        Field(name="Partner", dtype=String),
        # Whether customer has dependents
        Field(name="Dependents", dtype=String),
    ],
    online=True,   # make these features available in the online store for serving
    source=customer_stats_source,
    tags={"team": "churnguard", "model": "xgboost-churn"},
)