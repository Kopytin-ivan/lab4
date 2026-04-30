from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_RA_TOLERANCE = 3.2


def load_model_artifact(model_path: Path) -> dict:
    return joblib.load(model_path)


def add_model_predictions(features: pd.DataFrame, model_artifact: dict) -> pd.DataFrame:
    feature_cols = model_artifact["feature_columns"]
    model = model_artifact["model"]

    result = features.copy()
    result["predicted_ra"] = model.predict(result[feature_cols])
    result["prediction_error"] = result["roughness_ra"] - result["predicted_ra"]
    result["abs_prediction_error"] = result["prediction_error"].abs()
    return result


def add_quality_labels(features: pd.DataFrame, tolerance_ra: float = DEFAULT_RA_TOLERANCE) -> pd.DataFrame:
    result = features.copy()
    result["tolerance_ra"] = float(tolerance_ra)
    result["is_bad_quality"] = result["roughness_ra"] > tolerance_ra
    result["quality_label"] = np.where(result["is_bad_quality"], "bad", "ok")
    return result


def _threshold_candidates(values: pd.Series) -> np.ndarray:
    unique_values = np.sort(values.dropna().unique())
    if unique_values.size <= 1:
        return unique_values

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    quantiles = values.dropna().quantile(np.linspace(0.05, 0.95, 19)).to_numpy()
    return np.sort(np.unique(np.concatenate([midpoints, quantiles])))


def _threshold_metrics(y_true_bad: np.ndarray, y_pred_bad: np.ndarray) -> dict[str, float]:
    tp = float(np.sum(y_true_bad & y_pred_bad))
    tn = float(np.sum(~y_true_bad & ~y_pred_bad))
    fp = float(np.sum(~y_true_bad & y_pred_bad))
    fn = float(np.sum(y_true_bad & ~y_pred_bad))

    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    balanced_accuracy = (tpr + tnr) / 2

    return {
        "balanced_accuracy": balanced_accuracy,
        "precision_bad": precision,
        "recall_bad": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(
    features: pd.DataFrame,
    feature_col: str,
    tolerance_ra: float = DEFAULT_RA_TOLERANCE,
    higher_is_worse: bool = True,
) -> dict[str, float | str]:
    values = features[feature_col].astype(float)
    y_true_bad = (features["roughness_ra"] > tolerance_ra).to_numpy()
    candidates = _threshold_candidates(values)

    best_row = None
    for threshold in candidates:
        if higher_is_worse:
            y_pred_bad = (values >= threshold).to_numpy()
            direction = ">="
        else:
            y_pred_bad = (values <= threshold).to_numpy()
            direction = "<="

        if y_pred_bad.all() or (~y_pred_bad).all():
            continue

        metrics = _threshold_metrics(y_true_bad, y_pred_bad)
        above_bad_rate = float(y_true_bad[y_pred_bad].mean()) if y_pred_bad.any() else 0.0
        below_bad_rate = float(y_true_bad[~y_pred_bad].mean()) if (~y_pred_bad).any() else 0.0
        row = {
            "feature": feature_col,
            "direction": direction,
            "threshold": float(threshold),
            "bad_rate_after_threshold": above_bad_rate,
            "bad_rate_before_threshold": below_bad_rate,
            "bad_rate_lift": above_bad_rate - below_bad_rate,
            "samples_after_threshold": int(y_pred_bad.sum()),
            "samples_before_threshold": int((~y_pred_bad).sum()),
            **metrics,
        }

        if best_row is None:
            best_row = row
            continue

        best_key = (
            row["balanced_accuracy"],
            row["bad_rate_lift"],
            row["recall_bad"],
            -abs(row["samples_after_threshold"] - row["samples_before_threshold"]),
        )
        current_key = (
            best_row["balanced_accuracy"],
            best_row["bad_rate_lift"],
            best_row["recall_bad"],
            -abs(best_row["samples_after_threshold"] - best_row["samples_before_threshold"]),
        )
        if best_key > current_key:
            best_row = row

    if best_row is None:
        raise ValueError(f"Could not find a usable threshold for feature {feature_col}")

    return best_row


def threshold_report(
    features: pd.DataFrame,
    feature_cols: list[str],
    tolerance_ra: float = DEFAULT_RA_TOLERANCE,
) -> pd.DataFrame:
    rows = []
    for feature_col in feature_cols:
        correlation = features[[feature_col, "roughness_ra"]].corr().iloc[0, 1]
        higher_is_worse = bool(correlation >= 0)
        row = find_best_threshold(
            features=features,
            feature_col=feature_col,
            tolerance_ra=tolerance_ra,
            higher_is_worse=higher_is_worse,
        )
        row["corr_with_roughness_ra"] = float(correlation)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["balanced_accuracy", "bad_rate_lift"], ascending=False)
        .reset_index(drop=True)
    )


def condition_prediction_summary(predicted_features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predicted_features.groupby(["condition_id", "feed_mm", "speed_rpm", "depth_mm"], as_index=False)
        .agg(
            runs_count=("run_id", "count"),
            roughness_ra=("roughness_ra", "first"),
            predicted_ra_mean=("predicted_ra", "mean"),
            predicted_ra_std=("predicted_ra", "std"),
            xml_rms_mean=("xml_rms_mean", "mean"),
            sig_rms_mean=("sig_rms", "mean"),
            spec_power_2000_8000_mean=("spec_power_2000_8000", "mean"),
        )
        .sort_values(["predicted_ra_mean", "roughness_ra"])
        .reset_index(drop=True)
    )
    summary["productivity_proxy"] = summary["feed_mm"] * summary["speed_rpm"] * summary["depth_mm"]
    return summary


def recommend_modes(
    condition_summary: pd.DataFrame,
    target_ra: float = DEFAULT_RA_TOLERANCE,
    use_prediction: bool = True,
) -> pd.DataFrame:
    score_col = "predicted_ra_mean" if use_prediction else "roughness_ra"
    acceptable = condition_summary[condition_summary[score_col] <= target_ra].copy()
    acceptable["target_ra"] = float(target_ra)
    acceptable["margin_to_target"] = target_ra - acceptable[score_col]

    return (
        acceptable.sort_values(
            ["productivity_proxy", "margin_to_target", score_col],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
