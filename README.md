# F1_data_analysis
This repository contains the data analysis of F1 among different aspect of F1 race. 

# Professional Workflow Added

The repository now also contains a reusable analysis and machine-learning workflow built on top of the original notebook.

## Project Structure

- `F1_data_analysis.ipynb`: original exploratory notebook.
- `notebooks/02_professional_f1_analysis_and_ml.ipynb`: structured notebook with cell-by-cell explanations, ML mathematics, and pipeline usage.
- `src/f1_analysis/`: reusable code for loading data, engineering features, and training models.
- `scripts/run_f1_analysis_pipeline.py`: creates processed datasets and saved model artifacts.
- `scripts/generate_professional_notebook.py`: generates the teaching notebook.
- `data/interim/`: joined pit-stop analysis dataset.
- `data/processed/`: modeling dataset and dataset summary.
- `models/`: saved pipeline-ready estimators, metrics, and feature importances.
- `reports/`: written summary and figures.

## How To Run

1. Create or activate a Python environment.
2. Install dependencies from `requirements.txt`.
3. Run:

```bash
PYTHONPATH=src python scripts/run_f1_analysis_pipeline.py
PYTHONPATH=src python scripts/generate_professional_notebook.py
```

The saved models can later be loaded with `joblib` and used directly as preprocessing-plus-model pipelines.

# Key Findings: 

# 1
From 2011 to 2017, Toro Rosso had the most pitStop duration which was 31.207 seconds in 2015 in Singapore Grand Prix. He started at 8 and finished 8. McLaren had the minimum pitStop duration which was 13.173 with Lewis Hamilton in 2011 ( Hungarian Grand Prix). He started at 2 but suprisingly finished 4. 

# 2
There is almost no corelation between no.of pitStop and the position. Also there is no corelation between pitStop duration and the position. 

# 3
![image](https://github.com/user-attachments/assets/f9664e86-20c1-4340-9dec-90396688b8a2)
PLOT ANALYSIS : Plot shows the yearly average pit Stop Duration by Team from 2011 to 2017. There are both common and individualistic change among the teams. Mercedes has had the best performance in this time period averaging 21.953 seconds among all teams. Meanwhile, Ferrari has had the worst performance in 2015 averaging 24.258 seconds. Mercedes has the lowest average pitStop duration 
