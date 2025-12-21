"""
Orchestrator for quick end-to-end run: model selection + CV leaderboard.
This is a minimal bridge for the UI until full tuning/training wiring is added.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import pandas as pd
from sklearn.base import clone

from automl import models, selection, evaluation


def encode_classification_target(y: pd.Series) -> Tuple[pd.Series, List[str]]:
    """
    Factorize string/object labels into integer codes for model compatibility (e.g., XGBoost).
    Returns the encoded series and the list of original class labels by code index.
    """
    codes, uniques = pd.factorize(pd.Series(y).astype(str), sort=True)
    return pd.Series(codes, name=y.name), list(uniques)


def _drop_identifier_columns(df: pd.DataFrame, min_unique_ratio: float = 0.9) -> pd.DataFrame:
    """
    Remove obvious identifier/name-like columns and near-unique columns that don't help prediction.
    """
    to_drop: List[str] = []
    id_tokens = ["id", "identifier", "uuid", "guid", "serial", "name", "employee"]
    for col in df.columns:
        norm = str(col).strip().lower()
        if any(tok in norm for tok in id_tokens):
            to_drop.append(col)
            continue
        nunique = df[col].nunique(dropna=True)
        if len(df) and nunique >= min_unique_ratio * len(df):
            to_drop.append(col)
    return df.drop(columns=to_drop) if to_drop else df


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target]
    X = df.drop(columns=[target])
    X = _drop_identifier_columns(X)
    # Factorize non-numeric for a quick baseline
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col].astype(str))[0]
    return X, y


def run_quick_leaderboard(
    df: pd.DataFrame,
    target: str,
    task: str,
    n_categorical: int,
    cv_folds: int = 3,
) -> Tuple[List[evaluation.LeaderboardEntry], Optional[object], pd.DataFrame, pd.Series]:
    X, y = prepare_features(df, target)
    effective_task, _ = evaluation.infer_effective_task(task, y)
    label_classes: List[str] = []
    if effective_task == "classification":
        y, label_classes = encode_classification_target(y)
    candidate_specs = models.select_models(task=effective_task, n_rows=len(df), n_features=X.shape[1], n_categorical=n_categorical)
    candidate_models: List[Tuple[str, object]] = [(spec.name, spec.estimator) for spec in candidate_specs]
    leaderboard = evaluation.build_leaderboard(candidate_models, X, y, task=effective_task, cv_folds=cv_folds)

    # Fit best model on full data for downstream XAI/fairness use.
    best_model = None
    if leaderboard:
        best_name = leaderboard[0].model_name
        best_spec = next((s for s in candidate_specs if s.name == best_name), None)
        if best_spec:
            best_model = clone(best_spec.estimator)
            best_model.fit(X, y)
            if label_classes:
                # Store mapping for potential downstream decoding without changing return signature.
                setattr(best_model, "_label_classes", label_classes)

    return leaderboard, best_model, X, y
