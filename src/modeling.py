from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "roughness_ra"
GROUP_COL = "condition_id"
DROP_COLS = [GROUP_COL, "run_id", TARGET_COL]


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.columns
        if col not in DROP_COLS and pd.api.types.is_numeric_dtype(features[col])
    ]


def make_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=4,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=4,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def grouped_train_test_split(
    features: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = features[GROUP_COL]
    train_idx, test_idx = next(splitter.split(features, groups=groups))
    train = features.iloc[train_idx].copy().reset_index(drop=True)
    test = features.iloc[test_idx].copy().reset_index(drop=True)
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_models(
    features: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
    n_splits: int = 4,
) -> pd.DataFrame:
    X = features[feature_cols]
    y = features[TARGET_COL]
    groups = features[GROUP_COL]
    cv = GroupKFold(n_splits=n_splits)
    models = make_models(random_state=random_state)
    rows = []

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_MAE_mean": float(-scores["test_MAE"].mean()),
                "cv_MAE_std": float(scores["test_MAE"].std()),
                "cv_RMSE_mean": float(-scores["test_RMSE"].mean()),
                "cv_RMSE_std": float(scores["test_RMSE"].std()),
                "cv_R2_mean": float(scores["test_R2"].mean()),
                "cv_R2_std": float(scores["test_R2"].std()),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_MAE_mean").reset_index(drop=True)


def fit_and_evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    models = make_models(random_state=random_state)
    X_train = train[feature_cols]
    y_train = train[TARGET_COL].to_numpy()
    X_test = test[feature_cols]
    y_test = test[TARGET_COL].to_numpy()

    metric_rows = []
    prediction_frames = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_metrics = regression_metrics(y_train, train_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        metric_rows.append(
            {
                "model": name,
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
            }
        )

        pred_frame = test[[GROUP_COL, "run_id", "feed_mm", "speed_rpm", "depth_mm", TARGET_COL]].copy()
        pred_frame["model"] = name
        pred_frame["prediction"] = test_pred
        pred_frame["abs_error"] = np.abs(pred_frame[TARGET_COL] - pred_frame["prediction"])
        prediction_frames.append(pred_frame)

    metrics = pd.DataFrame(metric_rows).sort_values("test_MAE").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, fitted_models


def feature_importance_table(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model_artifacts(
    output_dir: Path,
    best_model_name: str,
    best_model: Pipeline,
    feature_cols: list[str],
    metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
        },
        output_dir / "best_roughness_model.joblib",
    )
    metrics.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8")
    cv_metrics.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
