from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "grid",
    "qualifying_position",
    "best_qualifying_seconds",
    "qualifying_delta_to_pole",
    "grid_minus_qualifying",
    "round",
    "lat",
    "lng",
    "alt",
    "driver_prev_standing_position",
    "driver_prev_season_points",
    "driver_prev_season_wins",
    "driver_avg_finish_last_5",
    "driver_avg_grid_last_5",
    "driver_avg_points_last_5",
    "driver_podium_rate_last_10",
    "driver_top10_rate_last_10",
    "driver_finish_rate_last_10",
    "constructor_avg_finish_last_5",
    "constructor_avg_grid_last_5",
    "constructor_avg_points_last_5",
    "constructor_podium_rate_last_10",
    "constructor_top10_rate_last_10",
    "constructor_finish_rate_last_10",
    "driver_constructor_pair_starts",
    "driver_circuit_starts",
    "constructor_circuit_starts",
    "driver_name",
    "constructor_name",
    "circuit_name",
    "circuit_country",
    "driver_nationality",
    "constructor_nationality",
]


@dataclass
class TrainedModelBundle:
    validation_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, dict[str, float]]
    chosen_models: dict[str, str]
    regression_model_path: str
    classification_model_path: str
    feature_importance_path: str
    metrics_path: str


def _build_preprocessor(features: list[str], frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = frame[features].select_dtypes(include="number").columns.tolist()
    categorical_features = [feature for feature in features if feature not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_df = df.dropna(subset=["qualifying_position"]).copy()
    return {
        "train": model_df[model_df["year"] <= 2015].copy(),
        "valid": model_df[model_df["year"] == 2016].copy(),
        "test": model_df[model_df["year"] == 2017].copy(),
        "train_valid": model_df[model_df["year"] <= 2016].copy(),
    }


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def _feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            importance = [0.0] * len(feature_names)
        else:
            importance = abs(coefficients.ravel())

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_select_models(df: pd.DataFrame, artifact_dir: Path) -> TrainedModelBundle:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    splits = _split_data(df)
    preprocessor = _build_preprocessor(FEATURE_COLUMNS, splits["train"])

    regression_candidates = {
        "ridge_finish_position": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))]
        ),
        "forest_finish_position": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    classification_candidates = {
        "logistic_podium": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                ),
            ]
        ),
        "forest_podium": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    validation_metrics: dict[str, dict[str, float]] = {}

    x_train = splits["train"][FEATURE_COLUMNS]
    x_valid = splits["valid"][FEATURE_COLUMNS]
    y_train_reg = splits["train"]["positionOrder"]
    y_valid_reg = splits["valid"]["positionOrder"]
    y_train_cls = splits["train"]["podium"]
    y_valid_cls = splits["valid"]["podium"]

    for name, pipeline in regression_candidates.items():
        pipeline.fit(x_train, y_train_reg)
        validation_metrics[name] = _regression_metrics(y_valid_reg, pipeline.predict(x_valid))

    for name, pipeline in classification_candidates.items():
        pipeline.fit(x_train, y_train_cls)
        valid_scores = pipeline.predict_proba(x_valid)[:, 1]
        validation_metrics[name] = _classification_metrics(
            y_valid_cls,
            pipeline.predict(x_valid),
            valid_scores,
        )

    chosen_regression = min(
        regression_candidates,
        key=lambda name: validation_metrics[name]["mae"],
    )
    chosen_classification = max(
        classification_candidates,
        key=lambda name: validation_metrics[name]["roc_auc"],
    )

    final_regression = regression_candidates[chosen_regression]
    final_classification = classification_candidates[chosen_classification]

    x_train_valid = splits["train_valid"][FEATURE_COLUMNS]
    y_train_valid_reg = splits["train_valid"]["positionOrder"]
    y_train_valid_cls = splits["train_valid"]["podium"]
    x_test = splits["test"][FEATURE_COLUMNS]
    y_test_reg = splits["test"]["positionOrder"]
    y_test_cls = splits["test"]["podium"]

    final_regression.fit(x_train_valid, y_train_valid_reg)
    final_classification.fit(x_train_valid, y_train_valid_cls)

    regression_predictions = final_regression.predict(x_test)
    classification_scores = final_classification.predict_proba(x_test)[:, 1]
    classification_predictions = final_classification.predict(x_test)

    test_metrics = {
        chosen_regression: _regression_metrics(y_test_reg, regression_predictions),
        chosen_classification: _classification_metrics(
            y_test_cls,
            classification_predictions,
            classification_scores,
        ),
    }

    regression_model_path = artifact_dir / "finish_position_pipeline.joblib"
    classification_model_path = artifact_dir / "podium_pipeline.joblib"
    metrics_path = artifact_dir / "model_metrics.json"
    feature_importance_path = artifact_dir / "podium_feature_importance.csv"

    joblib.dump(final_regression, regression_model_path)
    joblib.dump(final_classification, classification_model_path)

    feature_importance = _feature_importance_frame(final_classification)
    feature_importance.to_csv(feature_importance_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "chosen_models": {
                    "regression": chosen_regression,
                    "classification": chosen_classification,
                },
            },
            handle,
            indent=2,
        )

    return TrainedModelBundle(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        chosen_models={
            "regression": chosen_regression,
            "classification": chosen_classification,
        },
        regression_model_path=str(regression_model_path),
        classification_model_path=str(classification_model_path),
        feature_importance_path=str(feature_importance_path),
        metrics_path=str(metrics_path),
    )
