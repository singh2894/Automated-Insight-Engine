"""
Layer 4 – Feature Engineering.

Provides sklearn-style transformers for:
- Date part extraction.
- Categorical interaction hashes.
- Optional polynomial features (numeric).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class DatePartExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols: List[str], drop_original: bool = False):
        self.date_cols = date_cols
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in self.date_cols:
            if col not in df.columns:
                continue
            series = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = series.dt.year
            df[f"{col}_month"] = series.dt.month
            df[f"{col}_day"] = series.dt.day
            df[f"{col}_weekday"] = series.dt.weekday
            df[f"{col}_hour"] = series.dt.hour
            if self.drop_original:
                df = df.drop(columns=[col])
        return df


class CategoricalInteractionHasher(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols: List[str], n_pairs: int = 10, hash_space: int = 1024):
        self.cat_cols = cat_cols
        self.n_pairs = n_pairs
        self.hash_space = hash_space
        self.pairs_: List[tuple] = []

    def fit(self, X, y=None):
        # deterministically choose pairs based on ordering to keep reproducible
        cols = list(self.cat_cols)
        self.pairs_ = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                if len(self.pairs_) >= self.n_pairs:
                    return self
                self.pairs_.append((c1, c2))
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c1, c2 in self.pairs_:
            if c1 not in df.columns or c2 not in df.columns:
                continue
            combined = df[c1].astype(str) + "__" + df[c2].astype(str)
            # simple hash to reduce dimensionality
            hashed = combined.apply(lambda v: hash(v) % self.hash_space)
            df[f"{c1}__x__{c2}_h"] = hashed
        return df


class NumericPolyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols: List[str], degree: int = 2, include_bias: bool = False):
        self.numeric_cols = numeric_cols
        self.degree = degree
        self.include_bias = include_bias
        self.poly_: Optional[PolynomialFeatures] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.poly_.fit(df[self.numeric_cols])
        self.feature_names_ = self.poly_.get_feature_names_out(self.numeric_cols).tolist()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        poly_vals = self.poly_.transform(df[self.numeric_cols])
        poly_df = pd.DataFrame(poly_vals, columns=self.feature_names_, index=df.index)
        df = df.drop(columns=self.numeric_cols)
        df = pd.concat([df, poly_df], axis=1)
        return df
