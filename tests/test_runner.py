import pandas as pd

from automl import runner


def test_run_quick_leaderboard_regression():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    lb = runner.run_quick_leaderboard(df, target="y", task="regression", n_categorical=0, cv_folds=2)
    assert len(lb) >= 1
    assert lb[0].model_name in {"lin_reg", "rf_reg", "xgb_reg", "lgbm_reg", "svr"}
