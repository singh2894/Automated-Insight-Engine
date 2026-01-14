from AIE import models


def test_classification_registry_contains_core_models():
    reg = models.classification_registry()
    assert "log_reg" in reg and "rf" in reg and "xgb" in reg


def test_regression_registry_contains_core_models():
    reg = models.regression_registry()
    assert "lin_reg" in reg and "rf_reg" in reg and "xgb_reg" in reg


def test_selector_small_data_prefers_simple():
    selected = models.select_models(task="classification", n_rows=500, n_features=10, n_categorical=2)
    names = [m.name for m in selected]
    assert "xgb" not in names and "lgbm" not in names
    assert "log_reg" in names and "rf" in names


def test_selector_large_data_includes_boosters():
    selected = models.select_models(task="regression", n_rows=5000, n_features=50, n_categorical=5)
    names = [m.name for m in selected]
    assert "xgb_reg" in names and "lgbm_reg" in names
