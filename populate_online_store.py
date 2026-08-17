"""
ChurnGuard — Populate Feast Online Store
Writes features directly to the SQLite online store.
Run with: uv run python populate_online_store.py
"""

import pandas as pd
from feast import FeatureStore

# Load the parquet data
df = pd.read_parquet("feature_repo/data/customer_features.parquet")
print(f"Loaded {len(df)} rows")

# Initialize the feature store
store = FeatureStore(repo_path="feature_repo")

# Get the feature view
fv = store.get_feature_view("customer_features")

# Write directly to online store
store.write_to_online_store(
    feature_view_name="customer_features",
    df=df,
)

print(f"Written {len(df)} rows to online store")

# Verify by reading one customer back
result = store.get_online_features(
    features=[
        "customer_features:tenure",
        "customer_features:MonthlyCharges",
        "customer_features:Contract",
    ],
    entity_rows=[{"customerID": "7590-VHVEG"}],
).to_dict()

print("\nSample lookup for customer 7590-VHVEG:")
for k, v in result.items():
    print(f"  {k}: {v}")