from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


RAW_FILES = {
    "races": RAW_DATA_DIR / "races.csv",
    "results": RAW_DATA_DIR / "results.csv",
    "qualifying": RAW_DATA_DIR / "qualifying.csv",
    "drivers": RAW_DATA_DIR / "drivers.csv",
    "constructors": RAW_DATA_DIR / "constructors.csv",
    "circuits": RAW_DATA_DIR / "circuits.csv",
    "pitStops": RAW_DATA_DIR / "pitStops.csv",
    "driverStandings": RAW_DATA_DIR / "driverStandings.csv",
    "status": RAW_DATA_DIR / "status.csv",
}


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        NOTEBOOK_DIR,
        REPORT_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
