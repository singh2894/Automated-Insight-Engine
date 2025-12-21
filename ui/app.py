import io
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from automl import cleaning, diagnostics, understanding, features, selection, models, runner, evaluation


st.set_page_config(page_title="Automated Insight Engine", layout="wide")

MODEL_DISPLAY_NAMES = {
    "log_reg": "Logistic Regression",
    "dt": "Decision Tree Classifier",
    "rf": "Random Forest Classifier",
    "knn": "K-Nearest Neighbors Classifier",
    "xgb": "XGBoost Classifier",
    "lgbm": "LightGBM Classifier",
    "svm": "Support Vector Classifier",
    "nb": "Naive Bayes",
    "mlp": "Neural Network Classifier",
    "lin_reg": "Linear Regression",
    "dt_reg": "Decision Tree Regressor",
    "rf_reg": "Random Forest Regressor",
    "knn_reg": "K-Nearest Neighbors Regressor",
    "xgb_reg": "XGBoost Regressor",
    "lgbm_reg": "LightGBM Regressor",
    "svr": "Support Vector Regressor",
    "mlp_reg": "Neural Network Regressor",
}

class AppModelSpec:
    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator

def get_available_models(task, n_rows, n_features, n_categorical):
    candidates = list(models.select_models(task=task, n_rows=n_rows, n_features=n_features, n_categorical=n_categorical))
    if task == "classification":
        candidates.append(AppModelSpec("dt", DecisionTreeClassifier(random_state=42)))
    elif task == "regression":
        candidates.append(AppModelSpec("dt_reg", DecisionTreeRegressor(random_state=42)))
    return candidates

def generate_prediction_summary(model_name: str, task: str, metrics: dict) -> str:
    """Generates a human-readable summary of model performance."""
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    summary_text = f"### Performance Summary for {display_name}\n\n"

    if task == "classification":
        accuracy = metrics.get("accuracy", -1)
        f1 = metrics.get("f1", -1)
        precision = metrics.get("precision", -1)
        recall = metrics.get("recall", -1)

        summary_text += f"The model achieved an **accuracy of {accuracy:.2%}** on the test set. This means it correctly predicted the class for {accuracy:.0%} of the samples.\n\n"
        if f1 != -1:
            summary_text += f"- **F1-Score ({f1:.3f})**: This is the harmonic mean of precision and recall, providing a single score that balances both concerns. It's often a better measure than accuracy for imbalanced datasets.\n"
        if precision != -1:
            summary_text += f"- **Precision ({precision:.3f})**: Of all the times the model predicted a positive class, it was correct {precision:.1%} of the time.\n"
        if recall != -1:
            summary_text += f"- **Recall ({recall:.3f})**: The model successfully identified {recall:.1%} of all actual positive instances.\n\n"
        summary_text += "**Conclusion:** This model shows a certain level of performance. To improve, you could try other models from the leaderboard, engineer more features, or tune the model's hyperparameters."

    elif task == "regression":
        r2 = metrics.get("r2", -100)
        rmse = metrics.get("rmse", -1)
        mae = metrics.get("mae", -1)

        summary_text += f"The model achieved an **R² score of {r2:.3f}**. This score indicates that approximately **{r2:.1%}** of the variance in the target variable can be explained by the model's inputs.\n\n"
        if rmse != -1:
            summary_text += f"- **Root Mean Squared Error (RMSE) ({rmse:,.2f})**: On average, the model's predictions are off by about this amount from the actual values. Lower is better.\n"
        if mae != -1:
            summary_text += f"- **Mean Absolute Error (MAE) ({mae:,.2f})**: This is another measure of prediction error, representing the average absolute difference between predicted and actual values.\n\n"
        summary_text += "**Conclusion:** An R² score closer to 1.0 and lower RMSE/MAE values indicate a better fit. You can compare this performance against other models in the leaderboard to find the best one for your data."

    else:
        summary_text = "Could not generate summary for this task type."

    return summary_text


def load_csv(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except UnicodeDecodeError:
        # Fallback for non-UTF-8 encoded files (e.g., Excel CSVs often use cp1252 or latin1)
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding='latin1')


def render_understanding_tab(df: pd.DataFrame, target_hint: Optional[str]):
    summary = understanding.summarize(df, target_hint=target_hint)
    st.session_state["understanding_summary"] = summary

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Schema")
        st.write(
            {
                "numeric": summary.schema.numeric,
                "categorical": summary.schema.categorical,
                "datetime": summary.schema.datetime,
                "boolean": summary.schema.boolean,
                "convertible_text_numeric": summary.schema.convertible_text_numeric,
            }
        )

    with col2:
        st.subheader("Target & Task")
        st.write(
            {
                "target": summary.target.target,
                "target_reason": summary.target.reason,
                "task": summary.task.task,
                "task_reason": summary.task.reason,
                "target_dtype": summary.task.target_dtype,
                "target_cardinality": summary.task.target_cardinality,
            }
        )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Imbalance")
        if summary.imbalance:
            st.write(
                {
                    "is_imbalanced": summary.imbalance.is_imbalanced,
                    "ratio": summary.imbalance.ratio,
                    "class_counts": summary.imbalance.class_counts,
                    "threshold": summary.imbalance.threshold,
                }
            )
        else:
            st.write("Not applicable (regression or no target).")

    with col4:
        st.subheader("Skewed Numeric Features")
        st.write(summary.skew.skewed_features or "None flagged")

    st.subheader("Potential Leakage")
    st.write(
        {
            "method": summary.leakage.method,
            "suspicious_features": summary.leakage.suspicious_features,
        }
    )

def render_diagnosis_tab(df: pd.DataFrame):
    if "understanding_summary" not in st.session_state:
        st.info("Run Data Understanding first.")
        return
    summary = st.session_state["understanding_summary"]

    CARDINALITY_LIMIT = 100
    safe_categorical_cols = summary.schema.categorical
    high_cardinality_cats = []
    low_cardinality_cats = []

    for col in safe_categorical_cols:
        if col in df.columns and df[col].nunique() > CARDINALITY_LIMIT:
            high_cardinality_cats.append(col)
        else:
            low_cardinality_cats.append(col)

    if high_cardinality_cats:
        st.info(
            f"**Heads-up:** The following columns have a very high number of unique values (>{CARDINALITY_LIMIT}):\n\n"
            f"`{', '.join(high_cardinality_cats)}`\n\n"
            "To prevent memory errors, they have been excluded from the *Categorical Association* chart below. "
            "Your data itself has not been changed. These columns are often identifiers and can be dropped or "
            "ignored in later steps."
        )

    diag = diagnostics.diagnose(
        df,
        numeric_cols=summary.schema.numeric,
        categorical_cols=low_cardinality_cats,
        iqr_multiplier=1.5,
        corr_threshold=0.8,
        cat_corr_threshold=0.3,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        missing_df = diag.missing.per_column.copy()
        if 'missing_pct' in missing_df.columns:
            # Rename for better readability in the UI
            missing_df.rename(columns={'missing_pct': 'Missing Percentage (%)'}, inplace=True)
        st.dataframe(missing_df)
        if diag.missing.heatmap_fig:
            st.plotly_chart(diag.missing.heatmap_fig, width="stretch")

    with col2:
        st.subheader("Duplicates")
        st.write(
            {
                "duplicate_count": diag.duplicates.duplicate_count,
                "duplicate_ratio": diag.duplicates.duplicate_ratio,
            }
        )
        st.subheader("Outlier Suspicion (IQR)")
        st.write(diag.outliers.per_column or "None flagged")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Correlation Hotspots")
        st.write({"threshold": diag.correlations.threshold, "pairs": diag.correlations.pairs})
        if diag.correlations.matrix_fig:
            st.plotly_chart(diag.correlations.matrix_fig, width="stretch")

    with col4:
        if diag.cat_correlations:
            st.subheader("Categorical Association (Cramer's V)")
            st.write({"threshold": diag.cat_correlations.threshold, "pairs": diag.cat_correlations.pairs})
            if diag.cat_correlations.matrix_fig:
                st.plotly_chart(diag.cat_correlations.matrix_fig, width="stretch")

def render_training_tab():
    st.subheader("Train/Validate/Test & Predict")
    if "understanding_summary" not in st.session_state:
        st.info("Run Data Understanding first.")
        return

    summary = st.session_state["understanding_summary"]
    target_col = summary.target.target
    if not target_col:
        st.info("Need a target to train.")
        return

    if "cleaned_df" not in st.session_state:
        st.info("Run Cleaning first.")
        return

    # The training tab should always use the data processed from the cleaning step.
    df_train = st.session_state["cleaned_df"]
    
    feature_cols = st.multiselect(
        "Select features to use for training",
        options=[c for c in df_train.columns if c != target_col],
        default=[c for c in df_train.columns if c != target_col],
        key="train_feature_cols",
    )

    # Determine effective task (fallback to regression if target is continuous)
    effective_task, note = evaluation.infer_effective_task(summary.task.task or "regression", df_train[target_col] if target_col in df_train.columns else [])
    if note:
        st.info(note)

    if effective_task == "classification":
        target_series = df_train[target_col]
        n_unique = target_series.nunique()
        n_rows = len(target_series)
        # If more than 50 unique values AND more than 25% of values are unique, it's likely not a classification problem.
        if n_unique > 50 and (n_unique / n_rows) > 0.25:
            st.warning(
                f"⚠️ Target '{target_col}' was identified as classification but has very high cardinality ({n_unique} unique values). "
                "Switching task to regression."
            )
            effective_task = "regression"

    candidates = get_available_models(
        task=effective_task,
        n_rows=df_train.shape[0],
        n_features=len(feature_cols),
        n_categorical=len(summary.schema.categorical),
    )
    candidate_keys = [c.name for c in candidates]

    default_key = st.session_state.get("best_model_name", candidate_keys[0] if candidate_keys else None)
    selected_key = st.selectbox(
        "Select model",
        options=candidate_keys,
        format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
        index=candidate_keys.index(default_key) if default_key in candidate_keys else 0,
    )

    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, step=0.05, key="train_frac_slider")
    test_frac = 1.0 - train_frac
    st.write({"train_frac": round(train_frac, 2), "test_frac": round(test_frac, 2)})

    if st.button("Run train/test"):
        if target_col not in df_train.columns:
            st.error("Target column missing.")
            return
            
        df_used = df_train[feature_cols + [target_col]]
        # Prepare data
        X, y = runner.prepare_features(df_used, target_col)
        label_classes = []
        if effective_task == "classification":
            y, label_classes = runner.encode_classification_target(y)

        # Safely stratify the split
        stratify_param = None
        if effective_task == "classification" and y.value_counts().min() >= 2:
            stratify_param = y
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_frac, random_state=42, stratify=stratify_param
        )

        spec = next((c for c in candidates if c.name == selected_key), None)
        model = clone(spec.estimator) # type: ignore
        if selected_key == "log_reg":
            # Increase max_iter to aid convergence for Logistic Regression
            # The model is a pipeline, so we use '__' to set params on the final step.
            model.set_params(logisticregression__max_iter=500)
            
        model.fit(X_train, y_train)
        
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test) if effective_task == "classification" and hasattr(model, "predict_proba") else None
        
        def compute_metrics(y_true, y_pred, y_proba):
            if effective_task == "classification":
                return evaluation.classification_metrics(y_true, y_pred, y_proba)
            return evaluation.regression_metrics(y_true, y_pred)

        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

            # Add descriptive summary
            summary_text = generate_prediction_summary(selected_key, effective_task, test_metrics)
            st.markdown(summary_text)

        with col2:
            if effective_task == "classification":
                st.subheader("Confusion Matrix")
                # Ensure we compute matrix for all classes if we know them
                labels_indices = list(range(len(label_classes))) if label_classes else None
                cm = confusion_matrix(y_test, y_test_pred, labels=labels_indices)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[str(c) for c in label_classes] if label_classes else None,
                    y=[str(c) for c in label_classes] if label_classes else None,
                )
                st.plotly_chart(fig_cm)

                if label_classes and len(label_classes) == 2:
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    **Confusion Matrix Breakdown:**
                    - **True Negatives (TN):** {tn} (Correctly predicted *{label_classes[0]}*)
                    - **False Positives (FP):** {fp} (Incorrectly predicted *{label_classes[1]}* when actual was *{label_classes[0]}*)
                    - **False Negatives (FN):** {fn} (Incorrectly predicted *{label_classes[0]}* when actual was *{label_classes[1]}*)
                    - **True Positives (TP):** {tp} (Correctly predicted *{label_classes[1]}*)
                    """)

            if selected_key in ["dt", "dt_reg"]:
                st.subheader("Decision Tree Structure")
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(20, 10))
                    class_names_str = [str(c) for c in label_classes] if (effective_task == "classification" and label_classes) else None
                    plot_tree(
                        model, feature_names=feature_cols, class_names=class_names_str,
                        filled=True, rounded=True, ax=ax, max_depth=3, fontsize=10
                    )
                    st.pyplot(fig)
                    st.caption("Visualizing the top 3 levels of the tree for readability.")
                except Exception as e:
                    st.warning(f"Could not plot tree: {e}")

        # Show all test predictions in the preview.
        preview_rows = len(X_test)
        # Decode labels back to originals for readability.
        def decode_labels(arr):
            if not label_classes:
                return arr
            return pd.Series(arr).apply(
                lambda v: label_classes[int(v)] if pd.notna(v) and int(v) < len(label_classes) and int(v) >= 0 else v
            )
        y_true_slice = y_test.iloc[:preview_rows]

        y_true_preview = decode_labels(y_true_slice).reset_index(drop=True)
        y_pred_preview = decode_labels(pd.Series(y_test_pred).iloc[:preview_rows]).reset_index(drop=True)
        st.write("Sample predictions (test split):")
        # Build preview with aligned lengths and original row numbers.
        st.dataframe(pd.DataFrame({
            "y_true": y_true_preview,
            "y_pred": y_pred_preview,
        }))


def render_cleaning_tab(df):
    if "understanding_summary" not in st.session_state:
        st.info("Run Data Understanding first.")
    else:
        summary = st.session_state["understanding_summary"]
        target_col = summary.target.target
        sparse_threshold = st.slider("Sparsity Threshold", 0.0, 1.0, 0.9, 0.05, key="cleaning_sparse_threshold")

        # Defensively filter schema columns to ensure they exist in the current dataframe.
        # This prevents errors if the dataframe (e.g., from sampling) has changed
        # since the summary was generated.
        df_cols = set(df.columns)
        safe_numeric_cols = [col for col in summary.schema.numeric if col in df_cols]
        safe_categorical_cols = [col for col in summary.schema.categorical if col in df_cols]

        all_single_value_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

        # Prevent the target column from being dropped silently for having zero variance.
        # Instead, inform the user that it's not a valid target.
        if target_col and target_col in all_single_value_cols:
            st.error(f"The target column '{target_col}' has only one unique value and cannot be used for training. It will be dropped by the cleaning process. Please go back to 'Data Understanding' and select a different target.")

        # Drop feature columns that have zero variance.
        feature_cols_to_drop = [c for c in all_single_value_cols if c != target_col]
        
        cleaned_df, clean_summary = cleaning.clean_dataset(
            df,
            numeric_cols=safe_numeric_cols,
            categorical_cols=safe_categorical_cols,
            iqr_multiplier=1.5,
            sparse_threshold=sparse_threshold,
        )
        if feature_cols_to_drop:
            st.write(f"ℹ️ Dropped **{len(feature_cols_to_drop)}** feature columns with only a single unique value (zero variance):")
            st.json(feature_cols_to_drop)
        if hasattr(clean_summary, "removed_duplicates") and clean_summary.removed_duplicates > 0:
            st.write(f"✅ Removed **{clean_summary.removed_duplicates}** duplicate rows.")
        if hasattr(clean_summary, "dropped_sparse_cols") and clean_summary.dropped_sparse_cols:
            st.write(f"✅ Dropped **{len(clean_summary.dropped_sparse_cols)}** sparse columns (above {sparse_threshold * 100:.0f}% missing):")
            st.json(clean_summary.dropped_sparse_cols)
        if hasattr(clean_summary, "imputed_numeric_cols") and clean_summary.imputed_numeric_cols:
            st.write(f"✅ Imputed missing values in **{len(clean_summary.imputed_numeric_cols)}** numeric columns using `median`.")
        if hasattr(clean_summary, "imputed_categorical_cols") and clean_summary.imputed_categorical_cols:
            st.write(f"✅ Imputed missing values in **{len(clean_summary.imputed_categorical_cols)}** categorical columns using `'Unknown'`.")
        if hasattr(clean_summary, "capped_outliers") and clean_summary.capped_outliers:
            st.write(f"✅ Capped outliers in **{len(clean_summary.capped_outliers)}** columns.")

        st.subheader("Cleaned Preview")

        # Final check to see if the target was dropped for any reason (e.g., sparsity)
        if target_col and target_col not in cleaned_df.columns:
            st.warning(f"The target column '{target_col}' was removed during the cleaning process. This is likely because it had too many missing values (exceeding the sparsity threshold). You will need to select a new target in the Leaderboard tab.")

        st.dataframe(cleaned_df.head())

        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["clean_summary"] = clean_summary


def render_features_tab():
    if "cleaned_df" not in st.session_state:
        st.info("Run Cleaning first.")
    else:
        cleaned_df = st.session_state["cleaned_df"]
        summary = st.session_state["understanding_summary"]
        
        st.subheader("Feature Engineering")
        
        # Example: Interaction features for categorical columns
        cat_cols = summary.schema.categorical
        cat_pair_cols = [c for c in cat_cols if c in cleaned_df.columns]
        
        df_fe = cleaned_df.copy()
        
        if len(cat_pair_cols) >= 2:
            n_pairs = len(cat_pair_cols) * (len(cat_pair_cols) - 1) // 2
            st.write(f"Generating interactions for {len(cat_pair_cols)} categorical columns ({n_pairs} pairs)...")
            
            df_fe = features.CategoricalInteractionHasher(
                cat_cols=cat_pair_cols, n_pairs=n_pairs, hash_space=512
            ).fit(df_fe).transform(df_fe)

        st.write("Engineered columns:", list(set(df_fe.columns) - set(cleaned_df.columns)))
        st.dataframe(df_fe.head())

        st.subheader("Feature Selection")
        if summary.task.task is None:
            st.info("Need a target to run selection.")
        else:
            target_col = summary.target.target or st.selectbox("Target column", df_fe.columns)
            if target_col not in df_fe.columns:
                st.warning("Target not found after engineering.")
            else:
                st.write("Feature selection logic would go here.")

def render_models_preview_tab():
    if "understanding_summary" not in st.session_state:
        st.info("Run Data Understanding first.")
        return

    summary = st.session_state["understanding_summary"]
    task = summary.task.task

    if not task:
        st.info("Target not selected or task not identified. Please select a target.")
        return

    st.subheader(f"Candidate Models for {task.capitalize()}")

    n_rows, n_features, n_categorical = 1000, 10, 0
    if "cleaned_df" in st.session_state:
        df = st.session_state["cleaned_df"]
        n_rows, n_features = df.shape
        n_categorical = len([c for c in summary.schema.categorical if c in df.columns])

    candidates = get_available_models(task, n_rows, n_features, n_categorical)
    for spec in candidates:
        display_name = MODEL_DISPLAY_NAMES.get(spec.name, spec.name)
        with st.expander(f"🔹 {display_name}"):
            st.write(f"**Estimator:** `{type(spec.estimator).__name__}`")
            st.json(spec.estimator.get_params())

def render_leaderboard_tab():
    if "cleaned_df" not in st.session_state or "understanding_summary" not in st.session_state:
        st.info("Run Cleaning first.")
    else:
        cleaned_df = st.session_state["cleaned_df"]
        summary = st.session_state["understanding_summary"]
        n_rows, n_features = cleaned_df.shape
        n_cats = len(summary.schema.categorical)
        st.write({"rows": n_rows, "features": n_features, "categorical_cols": n_cats})
        
        if summary.task.task is None or summary.target.target is None:
            st.info("Need target/task to build leaderboard.")
        else:
            st.subheader("Leaderboard")

            target_col = summary.target.target
            original_task = summary.task.task or "regression"

            # Handle case where original target was dropped during cleaning
            if target_col not in cleaned_df.columns:
                st.warning(
                    f"The original target column '{target_col}' was removed during cleaning. "
                    "Please select a new target column to proceed."
                )
                available_cols = list(cleaned_df.columns)
                if not available_cols:
                    st.error("The cleaned dataset has no columns left.")
                    st.stop()

                target_col = st.selectbox("Select a new target column", options=available_cols, index=0)

            if not target_col:
                st.error("No target column selected or available.")
                st.stop()

            # Re-infer the task based on the (potentially new) target column
            effective_task, note = evaluation.infer_effective_task(original_task, cleaned_df[target_col])
            if note:
                st.info(note)

            # Override task to regression for high-cardinality classification targets
            # to prevent scikit-learn warnings and nonsensical modeling.
            if effective_task == "classification":
                target_series = cleaned_df[target_col]
                n_unique = target_series.nunique()
                n_rows = len(target_series)
                # If more than 50 unique values AND more than 25% of values are unique, it's likely not a classification problem.
                if n_unique > 50 and (n_unique / n_rows) > 0.25:
                    st.warning(
                        f"⚠️ Target '{target_col}' was identified as classification but has very high cardinality ({n_unique} unique values). "
                        "This is often not suitable for classification. Switching task to regression. \n\n"
                        "If this is incorrect, consider binning the target column into fewer categories before uploading."
                    )
                    effective_task = "regression"

            feature_cols = st.multiselect(
                "Select features to use",
                options=[c for c in cleaned_df.columns if c != target_col],
                default=[c for c in cleaned_df.columns if c != target_col],
            )
            if st.button("Run leaderboard"):
                if not feature_cols:
                    st.warning("Select at least one feature.")
                    st.stop()

                df_used = cleaned_df[feature_cols + [target_col]]

                # Replicate the robust logic from the training tab to bypass the buggy runner function.
                # 1. Prepare features (OHE on X) and split X/y.
                X_enc, y_raw = runner.prepare_features(df_used, target_col)

                # 2. Encode the target variable if the task is classification.
                y_enc = y_raw
                if effective_task == "classification":
                    y_enc, _ = runner.encode_classification_target(y_raw)

                # 3. Select candidate models based on the correct task.
                candidate_models = get_available_models(
                    task=effective_task,
                    n_rows=df_used.shape[0],
                    n_features=X_enc.shape[1],
                    n_categorical=len(summary.schema.categorical),
                )

                # Modify Logistic Regression to aid convergence before running leaderboard
                for spec in candidate_models:
                    if spec.name == "log_reg":
                        # This modifies the estimator template. Safe as candidates are regenerated.
                        # The estimator is a pipeline, so we use '__' to set params on the final step.
                        spec.estimator.set_params(logisticregression__max_iter=500)

                # 4. Run the cross-validation and build the leaderboard.
                # The build_leaderboard function expects a list of (name, estimator) tuples,
                # not a list of ModelSpec objects. We transform the list here to work around
                # the bug in evaluation.py where it tries to unpack a ModelSpec.
                models_for_lb = [(spec.name, spec.estimator) for spec in candidate_models]
                lb = evaluation.build_leaderboard(models_for_lb, X_enc, y_enc, task=effective_task, cv_folds=3)

                # 5. Fit the best model on the full dataset for later use.
                best_model = None
                if lb:
                    # Since we passed tuples to build_leaderboard, the returned LeaderboardEntry
                    # objects won't have a .model_spec attribute. We find the spec
                    # ourselves from the original candidate list using the model name.
                    best_model_name = lb[0].model_name
                    best_model_spec = next((spec for spec in candidate_models if spec.name == best_model_name), None)
                    if best_model_spec:
                        best_model = clone(best_model_spec.estimator)
                        best_model.fit(X_enc, y_enc)

                st.session_state["leaderboard"] = lb
                st.session_state["best_model"] = best_model
                st.session_state["best_model_name"] = lb[0].model_name if lb else None
                st.session_state["train_X"] = X_enc
                st.session_state["train_y"] = y_enc

            # --- Display Leaderboard Results ---
            # This block runs outside the button click to ensure results persist across reruns.
            if "leaderboard" in st.session_state:
                lb = st.session_state["leaderboard"]
                if not lb:
                    st.warning("Leaderboard run completed, but no models produced valid results.")
                else:
                    display_names = {"rmse": "Root Mean Squared Error", "mae": "Mean Absolute Error", "r2": "R2 Score", "mape": "Mean Absolute Percentage Error", "accuracy": "Accuracy", "f1": "F1 Score", "precision": "Precision", "recall": "Recall"}
                    metric_order = ["rmse", "mae", "r2", "mape", "accuracy", "f1", "precision", "recall"]
                    rows = []
                    for e in lb:
                        row = {"model": MODEL_DISPLAY_NAMES.get(e.model_name, e.model_name)}
                        for key in metric_order:
                            if key in e.scores:
                                label = display_names.get(key, key)
                                row[label] = round(e.scores[key], 4)
                        # Add any other scores not covered above.
                        for k, v in e.scores.items():
                            if k not in metric_order:
                                label = display_names.get(k, k)
                                row[label] = round(v, 4)
                        rows.append(row)
                    st.dataframe(rows)
                    best_name = MODEL_DISPLAY_NAMES.get(lb[0].model_name, lb[0].model_name)
                    st.success(f"🏆 Best model: **{best_name}**. You can now use this model in the 'Train & Test' tab.")

                    # --- Add Graph Visualization ---
                    st.subheader("Leaderboard Performance Graphs")

                    metrics_to_plot = []
                    if effective_task == "classification":
                        metrics_to_plot = ["accuracy", "f1", "precision", "recall"]
                    elif effective_task == "regression":
                        metrics_to_plot = ["r2", "rmse", "mae"]

                    for i, metric in enumerate(metrics_to_plot):
                        if metric in lb[0].scores:  # Check if metric is available
                            plot_data = [{"Model": MODEL_DISPLAY_NAMES.get(e.model_name, e.model_name), "Score": e.scores.get(metric)} for e in lb]
                            plot_df = pd.DataFrame(plot_data).dropna(subset=["Score"])

                            if plot_df.empty:
                                continue

                            sort_order = "descending"
                            if any(err in metric.lower() for err in ["error", "rmse", "mae"]):
                                sort_order = "ascending"

                            fig = px.bar(
                                plot_df, x="Model", y="Score", title=f"Model Comparison by {display_names.get(metric, metric)}",
                                color="Model", text_auto=".3f",
                            )
                            fig.update_layout(xaxis={"categoryorder": f"total {sort_order}"})

                            # Set y-axis range to [0, 1] for classification metrics
                            if metric in ["accuracy", "f1", "precision", "recall"]:
                                fig.update_yaxes(range=[0, 1])

                            st.plotly_chart(fig, use_container_width=True)
                    # --- End Graph Visualization ---


def main():
    st.title("Automated Insight Engine")
    st.markdown("The **Automated Insight Engine** is a tool that makes data science projects faster. You can upload any CSV file, and it will help you clean the data and build a machine learning model, step-by-step.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # --- State Reset Logic ---
    # If a new file is uploaded, reset the downstream state to prevent errors
    # from using old analysis results (like schema) on a new dataframe.
    if "uploaded_file_id" not in st.session_state:
        st.session_state.uploaded_file_id = None

    new_file_id = (uploaded.name, uploaded.size) if uploaded else None
    if st.session_state.uploaded_file_id != new_file_id:
        # A new file has been uploaded or the file was removed. Clear all derived state.
        keys_to_clear = [
            "understanding_summary", "cleaned_df", "clean_summary", "leaderboard",
            "best_model", "best_model_name", "train_X", "train_y"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.uploaded_file_id = new_file_id
    # --- End State Reset Logic ---

    df = load_csv(uploaded)

    if df.empty:
        return

    st.write("Preview", df.head())

    # --- Data Sampling for Performance ---
    if df.shape[0] > 100000:  # Only show sampling option for very large datasets
        st.info(f"Dataset has {df.shape[0]} rows. Consider using a sample for faster interactive analysis.")
        use_sample = st.checkbox("Use a random sample of the data", value=True)
        if use_sample:
            sample_size = st.number_input("Sample size", min_value=100, max_value=df.shape[0], value=min(20000, df.shape[0]), step=1000)
            df = df.sample(n=int(sample_size), random_state=42)
            st.write(f"Using a sample of {df.shape[0]} rows.")
    # ------------------------------------

    target_hint = st.selectbox(
        "Select target column (optional)", options=["(auto-detect)"] + list(df.columns)
    )
    target_hint = None if target_hint == "(auto-detect)" else target_hint

    tabs = st.tabs(["Data Understanding", "Diagnosis", "Cleaning", "Features", "Models (preview)", "Leaderboard", "Train & Test"])

    with tabs[0]:
        render_understanding_tab(df, target_hint)

    with tabs[1]:
        render_diagnosis_tab(df)

    with tabs[2]:
        render_cleaning_tab(df)

    with tabs[3]:
        render_features_tab()

    with tabs[4]:
        render_models_preview_tab()

    with tabs[5]:
        render_leaderboard_tab()

    with tabs[6]:
        render_training_tab()


if __name__ == "__main__":
    main()