# F1 Analysis Summary

## Scope

This summary extends the original notebook and separates the work into two tracks:

- pit-stop operational analysis
- pre-race outcome prediction

That split is important because pit-stop variables help explain race outcomes after the event, but they should not be used as pre-race predictors in a deployable forecasting pipeline.

## Dataset Built

- Pit-stop analysis dataset: `6251` rows
- Modeling dataset: `4165` rows
- Modeling window: `2008` to `2017`

## Best Machine Learning Tasks For This Dataset

1. Constructor pit-stop efficiency analysis across seasons.
2. Driver and constructor momentum analysis using rolling performance indicators.
3. Finish-position regression from qualifying, grid, and prior-form features.
4. Podium classification for race-entry level decision support.

## Saved Model Outcomes

- Regression winner: `ridge_finish_position`
- Regression holdout MAE: `3.554`
- Regression holdout RMSE: `4.605`
- Classification winner: `forest_podium`
- Classification holdout accuracy: `0.887`
- Classification holdout F1: `0.681`
- Classification holdout ROC-AUC: `0.943`

## Main Signals Learned By The Podium Model

Top features from the saved feature-importance table:

- qualifying position
- grid position
- qualifying delta to pole
- previous driver standing position
- recent driver podium rate
- recent constructor grid and points form

## Reusable Artifacts

- `models/finish_position_pipeline.joblib`
- `models/podium_pipeline.joblib`
- `models/model_metrics.json`
- `models/podium_feature_importance.csv`

These artifacts are pipeline-ready because preprocessing and model logic are stored together.
