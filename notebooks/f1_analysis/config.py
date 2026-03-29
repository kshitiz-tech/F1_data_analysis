from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


RAW_FILES = {
    "races": PROJECT_ROOT / "races.csv",
    "results": PROJECT_ROOT / "results.csv",
    "qualifying": PROJECT_ROOT / "qualifying.csv",
    "drivers": PROJECT_ROOT / "drivers.csv",
    "constructors": PROJECT_ROOT / "constructors.csv",
    "circuits": PROJECT_ROOT / "circuits.csv",
    "pitStops": PROJECT_ROOT / "pitStops.csv",
    "driverStandings": PROJECT_ROOT / "driverStandings.csv",
    "status": PROJECT_ROOT / "status.csv",
}


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        NOTEBOOK_DIR,
        REPORT_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
