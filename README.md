# F1 Data Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/Project-Organized-success)

Professional Formula 1 data analysis and machine learning on historical race results, qualifying sessions, pit stops, and standings.

## Overview

This repository combines:

- exploratory race and pit-stop analysis
- reusable feature engineering code
- pipeline-ready machine learning models
- a cleaner project structure for GitHub presentation

The project focuses on two separate analysis tracks:

1. Pit-stop operational analysis
2. Pre-race prediction of finish position and podium outcome

That split is intentional. Pit-stop variables are useful for explaining what happened in a race, but they should not be used as pre-race inputs for a deployable prediction pipeline.

## Key Results

- Finish-position regression holdout performance:
  - MAE: `3.554`
  - RMSE: `4.605`
- Podium classification holdout performance:
  - Accuracy: `0.887`
  - F1: `0.681`
  - ROC-AUC: `0.943`

Top podium-model signals:

- qualifying position
- grid position
- qualifying delta to pole
- previous driver standing position
- recent driver and constructor form

## Project Structure

```text
F1_data_analysis/
├── F1_data_analysis.ipynb
├── archive/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/f1_analysis/
├── .gitignore
├── README.md
└── requirements.txt
```

## Main Files

- `F1_data_analysis.ipynb`: original notebook-based analysis
- `notebooks/02_professional_f1_analysis_and_ml.ipynb`: structured notebook with explanations and ML mathematics
- `src/f1_analysis/`: reusable code for loading data, building features, and training models
- `scripts/run_f1_analysis_pipeline.py`: generates processed datasets and trained artifacts
- `scripts/generate_professional_notebook.py`: regenerates the structured notebook
- `reports/analysis_summary.md`: concise written summary

## Data Layout

- `data/raw/`: source CSV files
- `data/interim/`: joined analysis tables
- `data/processed/`: cleaned modeling dataset and dataset summary

## Saved Artifacts

- `models/finish_position_pipeline.joblib`
- `models/podium_pipeline.joblib`
- `models/model_metrics.json`
- `models/podium_feature_importance.csv`

These models are saved as full preprocessing-plus-model pipelines, so they can be reused later in scripts or applications without rebuilding the preprocessing steps manually.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_f1_analysis_pipeline.py
PYTHONPATH=src python scripts/generate_professional_notebook.py
```

## Cleanup Notes

- unrelated notebooks were moved into `archive/notebooks/`
- unrelated data files were moved into `archive/misc/`
- raw dataset files were moved from the repository root into `data/raw/`

This keeps the repository root focused on the actual F1 project rather than scratch files.
