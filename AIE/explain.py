"""
Layer 8 – Explainability utilities (SHAP-based).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import shap


def shap_feature_importance(model, X: pd.DataFrame, max_display: int = 20) -> Dict[str, float]:
    # Use TreeExplainer for tree models; fallback to LinearExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, shap_values can be list
        if isinstance(shap_values, list):
            shap_vals = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_vals = np.abs(shap_values).mean(axis=0)
    except Exception:
        explainer = shap.LinearExplainer(model, X)
        shap_vals = np.abs(explainer.shap_values(X)).mean(axis=0)

    feature_names = getattr(model, "feature_names_in_", X.columns)
    importance = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    # Return top features
    top = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:max_display])
    return top
