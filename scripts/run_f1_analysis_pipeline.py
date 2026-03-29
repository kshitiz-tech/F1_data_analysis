from __future__ import annotations

import json
import sys

import pandas as pd

from f1_analysis.config import INTERIM_DATA_DIR, MODEL_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from f1_analysis.dataset import build_modeling_dataset, build_pit_stop_dataset, load_core_tables
from f1_analysis.modeling import FEATURE_COLUMNS, train_and_select_models


def main() -> int:
    ensure_project_dirs()
    tables = load_core_tables()

    pit_stop_dataset = build_pit_stop_dataset(tables)
    modeling_dataset = build_modeling_dataset(tables)

    pit_stop_path = INTERIM_DATA_DIR / "pit_stop_analysis_dataset.csv"
    modeling_path = PROCESSED_DATA_DIR / "f1_modeling_dataset.csv"
    summary_path = PROCESSED_DATA_DIR / "dataset_summary.json"

    pit_stop_dataset.to_csv(pit_stop_path, index=False)
    modeling_dataset.to_csv(modeling_path, index=False)

    bundle = train_and_select_models(modeling_dataset, MODEL_DIR)

    summary = {
        "pit_stop_rows": int(len(pit_stop_dataset)),
        "modeling_rows": int(len(modeling_dataset)),
        "modeling_year_span": [
            int(modeling_dataset["year"].min()),
            int(modeling_dataset["year"].max()),
        ],
        "model_features": FEATURE_COLUMNS,
        "saved_models": bundle.chosen_models,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved:", pit_stop_path)
    print("Saved:", modeling_path)
    print("Saved:", summary_path)
    print("Saved:", bundle.regression_model_path)
    print("Saved:", bundle.classification_model_path)
    print("Saved:", bundle.feature_importance_path)
    print("Saved:", bundle.metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
