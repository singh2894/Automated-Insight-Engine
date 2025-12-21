"""
Layer 4 – Feature Selection.

Provides utilities to run filter/embedded/wrapper methods and combine feature importances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression, chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class SelectionResult:
    selected_features: List[str]
    importances: Dict[str, float]
    notes: List[str]


def _encode_non_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
    return X_enc


def _infer_effective_task(task: str, y: pd.Series) -> (str, str):
    """
    If task is set to classification but the target looks continuous, fall back to regression scorers.
    Returns the effective task and an optional note.
    """
    note = ""
    if task == "classification":
        y_series = pd.Series(y)
        if pd.api.types.is_numeric_dtype(y_series):
            unique_vals = y_series.nunique(dropna=True)
            is_float = pd.api.types.is_float_dtype(y_series)
            if is_float or unique_vals > max(20, 0.1 * len(y_series)):
                task = "regression"
                note = "Target appears continuous; using regression scorers instead of classification."
    return task, note


def filter_methods(X: pd.DataFrame, y: pd.Series, task: str, variance_threshold: float = 0.0) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}

    if variance_threshold > 0:
        vt = VarianceThreshold(threshold=variance_threshold)
        vt.fit(X)
        scores["variance"] = {col: float(vt.variances_[i]) for i, col in enumerate(X.columns)}

    X_enc = _encode_non_numeric(X)

    if task == "classification":
        try:
            mi = mutual_info_classif(X_enc, y, discrete_features="auto")
        except ValueError:
            # Fallback when the target isn't suitable for classification MI (e.g., too many unique values).
            mi = mutual_info_regression(X_enc, y, discrete_features="auto")
    else:
        mi = mutual_info_regression(X_enc, y, discrete_features="auto")
    scores["mutual_info"] = {col: float(mi[i]) for i, col in enumerate(X.columns)}

    # Chi-square only for non-negative features, mostly for categorical one-hot encoded
    if task == "classification":
        try:
            X_nonneg = X_enc.copy()
            X_nonneg[X_nonneg < 0] = 0
            chi_vals, _ = chi2(X_nonneg, y)
            scores["chi2"] = {col: float(chi_vals[i]) for i, col in enumerate(X.columns)}
        except ValueError:
            # If target isn't valid for chi-square, skip gracefully.
            scores["chi2"] = {}

    return scores


def embedded_methods(X: pd.DataFrame, y: pd.Series, task: str) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    if task == "classification":
        try:
            l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, tol=1e-3)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                l1.fit(X, y)
            scores["l1"] = {col: float(abs(coef)) for col, coef in zip(X.columns, l1.coef_[0])}
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores["tree"] = {col: float(imp) for col, imp in zip(X.columns, rf.feature_importances_)}
            return scores
        except ValueError:
            # If target is not suitable for classification, fall back to regression scorers.
            task = "regression"

    # Regression path (used directly or as fallback).
    l1 = Lasso(alpha=0.001, max_iter=5000, tol=1e-3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        l1.fit(X, y)
    scores["l1"] = {col: float(abs(coef)) for col, coef in zip(X.columns, l1.coef_)}
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    scores["tree"] = {col: float(imp) for col, imp in zip(X.columns, rf.feature_importances_)}
    return scores


def wrapper_rfe(X: pd.DataFrame, y: pd.Series, task: str, n_features_to_select: Optional[int] = None) -> Dict[str, float]:
    estimator = None
    if task == "classification":
        try:
            estimator = LogisticRegression(max_iter=500)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select or max(1, X.shape[1] // 2))
            rfe.fit(X, y)
            return {col: float(rank) for col, rank in zip(X.columns, rfe.ranking_)}
        except ValueError:
            # Fallback to regression-style RFE if classification target is invalid.
            task = "regression"

    estimator = Lasso(alpha=0.001, max_iter=2000)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select or max(1, X.shape[1] // 2))
    rfe.fit(X, y)
    return {col: float(rank) for col, rank in zip(X.columns, rfe.ranking_)}


def unify_importances(score_dicts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, List[float]] = {}
    for method_scores in score_dicts.values():
        for feat, score in method_scores.items():
            agg.setdefault(feat, []).append(score)
    return {feat: float(np.mean(scores)) for feat, scores in agg.items()}


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    variance_threshold: float = 0.0,
    top_k: Optional[int] = None,
) -> SelectionResult:
    notes: List[str] = []
    effective_task, note = _infer_effective_task(task, y)
    if note:
        notes.append(note)

    # Encode non-numeric upfront to avoid errors downstream
    X_proc = _encode_non_numeric(X)
    filter_scores = filter_methods(X, y, effective_task, variance_threshold=variance_threshold)
    embedded_scores = embedded_methods(X_proc, y, effective_task)
    wrapper_scores = wrapper_rfe(X_proc, y, effective_task)

    combined_scores = {**filter_scores, **embedded_scores, "rfe_rank": wrapper_scores}
    unified = unify_importances({k: v for k, v in combined_scores.items() if k != "rfe_rank"})

    # Lower rank is better in RFE; invert to comparable importance
    max_rank = max(wrapper_scores.values()) if wrapper_scores else 1
    rfe_importance = {k: (max_rank - v + 1) for k, v in wrapper_scores.items()}

    fused = {}
    for feat in X.columns:
        vals = []
        if feat in unified:
            vals.append(unified[feat])
        if feat in rfe_importance:
            vals.append(rfe_importance[feat])
        fused[feat] = float(np.mean(vals)) if vals else 0.0

    sorted_feats = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k:
        selected = [f for f, _ in sorted_feats[:top_k]]
    else:
        selected = [f for f, score in sorted_feats if score > 0]

    return SelectionResult(selected_features=selected, importances=fused, notes=notes)
