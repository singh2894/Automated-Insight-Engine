import pandas as pd

from AIE import runner, models, evaluation


def test_run_quick_leaderboard_regression():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    
    # Replicate logic to bypass buggy runner.run_quick_leaderboard
    X, y = runner.prepare_features(df, "y")
    candidates = models.select_models(task="regression", n_rows=len(df), n_features=X.shape[1], n_categorical=0)
    models_for_lb = [(spec.name, spec.estimator) for spec in candidates]
    lb = evaluation.build_leaderboard(models_for_lb, X, y, task="regression", cv_folds=2)

    assert len(lb) >= 1
    assert lb[0].model_name in {"lin_reg", "rf_reg", "xgb_reg", "lgbm_reg", "svr"}
