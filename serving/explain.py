"""
ChurnGuard — SHAP Explanation
Computes SHAP values for a prediction and returns the top contributing features.
These values are passed to the LLM explanation layer in Phase 5.
"""

import pandas as pd
import shap


def compute_shap_values(model, feature_df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Compute SHAP values for a single customer's features.

    Args:
        model: Trained XGBoost model
        feature_df: DataFrame with exactly one row — the customer's features
        top_n: How many top features to return

    Returns:
        Dictionary of {feature_name: shap_value} for the top N features,
        sorted by absolute impact (most influential first).
    """
    # TreeExplainer is the fastest SHAP method for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values — shape: (1 row, n_features)
    shap_values = explainer.shap_values(feature_df)

    # Get feature names and their SHAP values for this customer
    feature_names = feature_df.columns.tolist()
    shap_scores = shap_values[0]  # first (only) row

    # Pair feature names with their SHAP values
    shap_dict = dict(zip(feature_names, shap_scores))

    # Sort by absolute value — biggest impact first
    top_features = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    )

    # Round for readability
    top_features = {k: round(float(v), 4) for k, v in top_features.items()}

    return top_features


def shap_to_text(shap_dict: dict) -> str:
    """
    Convert SHAP values into a human-readable string.
    This string gets passed to the LLM in Phase 5 as context.

    Example output:
        Contract: +0.3821 (increases churn risk)
        tenure: -0.2134 (decreases churn risk)
    """
    lines = []
    for feature, value in shap_dict.items():
        direction = "increases churn risk" if value > 0 else "decreases churn risk"
        lines.append(f"  {feature}: {value:+.4f} ({direction})")
    return "\n".join(lines)