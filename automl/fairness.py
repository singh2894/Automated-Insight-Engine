"""
Layer 8 – Simple fairness checks.

Computes group-wise performance for sensitive columns and flags disparities.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error


def group_classification_metrics(y_true, y_pred, group: pd.Series) -> Dict[str, Dict[str, float]]:
    results = {}
    for g in group.dropna().unique():
        mask = group == g
        if mask.sum() == 0:
            continue
        results[g] = {
            "f1": f1_score(y_true[mask], y_pred[mask], average="binary"),
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
        }
    return results


def group_regression_metrics(y_true, y_pred, group: pd.Series) -> Dict[str, Dict[str, float]]:
    results = {}
    for g in group.dropna().unique():
        mask = group == g
        if mask.sum() == 0:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        results[g] = {"rmse": rmse}
    return results


def disparity_alerts(group_metrics: Dict[str, Dict[str, float]], metric: str, max_gap: float) -> List[str]:
    alerts = []
    values = [m[metric] for m in group_metrics.values() if metric in m]
    if not values:
        return alerts
    max_val = max(values)
    min_val = min(values)
    gap = max_val - min_val
    if gap > max_gap:
        alerts.append(f"Gap {gap:.3f} in {metric} exceeds {max_gap}")
    return alerts
