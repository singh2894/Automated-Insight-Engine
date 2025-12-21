import pandas as pd
import numpy as np

from automl import features as f
from automl import selection as s


def test_date_part_extractor_adds_parts():
    df = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3, freq="D")})
    ext = f.DatePartExtractor(date_cols=["d"], drop_original=True)
    out = ext.transform(df)
    assert "d_year" in out.columns and "d_month" in out.columns
    assert "d" not in out.columns


def test_cat_interaction_hasher_creates_pairs():
    df = pd.DataFrame({"c1": ["a", "b"], "c2": ["x", "y"], "c3": ["u", "v"]})
    hasher = f.CategoricalInteractionHasher(cat_cols=["c1", "c2", "c3"], n_pairs=2, hash_space=100)
    hasher.fit(df)
    out = hasher.transform(df)
    new_cols = [c for c in out.columns if "_h" in c]
    assert len(new_cols) == 2


def test_numeric_poly_features():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    poly = f.NumericPolyFeatures(numeric_cols=["x"], degree=2, include_bias=False)
    poly.fit(df)
    out = poly.transform(df)
    assert "x^2" in poly.feature_names_[1]  # ensure poly term exists
    assert out.shape[1] > df.shape[1] - 1  # replaced x with poly features


def test_selection_pipeline_classification():
    df = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [4, 3, 2, 1], "x3": [1, 1, 1, 1]})
    y = pd.Series([0, 0, 1, 1])
    result = s.select_features(df, y, task="classification", variance_threshold=0.0, top_k=2)
    assert len(result.selected_features) == 2
    assert "x3" not in result.selected_features  # constant feature unlikely to rank


def test_selection_pipeline_regression():
    df = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])
    result = s.select_features(df, y, task="regression", variance_threshold=0.0, top_k=1)
    assert len(result.selected_features) == 1
