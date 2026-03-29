from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_FILES


def read_csv_with_fallback(path: str | Any, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def time_to_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return np.nan


def load_core_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "races": read_csv_with_fallback(RAW_FILES["races"], parse_dates=["date"]),
        "results": read_csv_with_fallback(RAW_FILES["results"]),
        "qualifying": read_csv_with_fallback(RAW_FILES["qualifying"]),
        "drivers": read_csv_with_fallback(RAW_FILES["drivers"]).rename(
            columns={"nationality": "driver_nationality"}
        ),
        "constructors": read_csv_with_fallback(RAW_FILES["constructors"]).drop(
            columns=["Unnamed: 5"], errors="ignore"
        ),
        "circuits": read_csv_with_fallback(RAW_FILES["circuits"]),
        "pitStops": read_csv_with_fallback(RAW_FILES["pitStops"]),
        "driverStandings": read_csv_with_fallback(RAW_FILES["driverStandings"]),
        "status": read_csv_with_fallback(RAW_FILES["status"]),
    }
    return tables


def build_pit_stop_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    races = tables["races"]
    pit_stops = tables["pitStops"].copy()
    results = tables["results"][["raceId", "driverId", "constructorId", "positionOrder", "grid"]]
    drivers = tables["drivers"][["driverId", "forename", "surname"]].copy()
    constructors = tables["constructors"][["constructorId", "name"]].rename(
        columns={"name": "constructor_name"}
    )

    drivers["driver_name"] = drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")

    pit_stop_dataset = (
        pit_stops.merge(races[["raceId", "year", "round", "name", "date"]], on="raceId", how="left")
        .merge(results, on=["raceId", "driverId"], how="left")
        .merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
        .merge(constructors, on="constructorId", how="left")
    )

    return pit_stop_dataset.sort_values(["date", "raceId", "driverId", "stop"]).reset_index(drop=True)


def _add_shifted_rolling_features(df: pd.DataFrame, group_key: str, prefix: str) -> pd.DataFrame:
    grouped = df.groupby(group_key, group_keys=False)

    df[f"{prefix}_avg_finish_last_5"] = grouped["positionOrder"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    df[f"{prefix}_avg_grid_last_5"] = grouped["grid"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    df[f"{prefix}_avg_points_last_5"] = grouped["race_points"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    df[f"{prefix}_podium_rate_last_10"] = grouped["podium"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )
    df[f"{prefix}_top10_rate_last_10"] = grouped["top10"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )
    df[f"{prefix}_finish_rate_last_10"] = grouped["finished_clean"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )

    return df


def build_modeling_dataset(
    tables: dict[str, pd.DataFrame],
    start_year: int = 2008,
    end_year: int = 2017,
) -> pd.DataFrame:
    races = tables["races"]
    results = tables["results"].rename(columns={"points": "race_points"}).copy()
    qualifying = tables["qualifying"].copy()
    drivers = tables["drivers"]
    constructors = tables["constructors"]
    circuits = tables["circuits"]
    standings = tables["driverStandings"].copy()
    status = tables["status"]

    for session in ("q1", "q2", "q3"):
        qualifying[f"{session}_seconds"] = qualifying[session].map(time_to_seconds)
    qualifying["best_qualifying_seconds"] = qualifying[
        ["q1_seconds", "q2_seconds", "q3_seconds"]
    ].min(axis=1)

    race_window = races[races["year"].between(start_year, end_year)].copy()

    df = results.merge(
        race_window[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId",
        how="inner",
    )
    df = df.merge(
        qualifying[["raceId", "driverId", "position", "best_qualifying_seconds"]],
        on=["raceId", "driverId"],
        how="left",
        suffixes=("", "_qualifying"),
    ).rename(columns={"position_qualifying": "qualifying_position"})
    df = df.merge(
        drivers[["driverId", "driverRef", "forename", "surname", "driver_nationality"]],
        on="driverId",
        how="left",
    )
    df["driver_name"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
    df = df.merge(
        constructors[["constructorId", "name", "nationality"]].rename(
            columns={"name": "constructor_name", "nationality": "constructor_nationality"}
        ),
        on="constructorId",
        how="left",
    )
    df = df.merge(
        circuits[["circuitId", "name", "country", "lat", "lng", "alt"]].rename(
            columns={"name": "circuit_name", "country": "circuit_country"}
        ),
        on="circuitId",
        how="left",
    )
    df = df.merge(status, on="statusId", how="left")

    standings_subset = standings[["raceId", "driverId", "points", "wins", "position"]].rename(
        columns={
            "points": "standing_points_after_race",
            "wins": "standing_wins_after_race",
            "position": "standing_position_after_race",
        }
    )
    df = df.merge(standings_subset, on=["raceId", "driverId"], how="left")

    df = df.sort_values(["date", "raceId", "positionOrder", "driverId"]).reset_index(drop=True)

    df["podium"] = (df["positionOrder"] <= 3).astype(int)
    df["top10"] = (df["positionOrder"] <= 10).astype(int)
    df["win"] = (df["positionOrder"] == 1).astype(int)
    df["finished_clean"] = (
        df["status"].fillna("").str.contains(r"Finished|\+", regex=True).astype(int)
    )

    df["qualifying_delta_to_pole"] = df.groupby("raceId")["best_qualifying_seconds"].transform(
        lambda series: series - series.min()
    )
    df["grid_minus_qualifying"] = df["grid"] - df["qualifying_position"]

    df = df.sort_values(["driverId", "date", "raceId"]).reset_index(drop=True)
    df["driver_prev_standing_position"] = df.groupby("driverId")[
        "standing_position_after_race"
    ].shift(1)
    df["driver_prev_season_points"] = (
        df.groupby(["driverId", "year"])["race_points"].cumsum() - df["race_points"]
    )
    df["driver_prev_season_wins"] = (
        df.groupby(["driverId", "year"])["win"].cumsum() - df["win"]
    )

    df = _add_shifted_rolling_features(df, "driverId", "driver")
    df = _add_shifted_rolling_features(df, "constructorId", "constructor")

    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)
    df["driver_constructor_pair_starts"] = df.groupby(["driverId", "constructorId"]).cumcount()
    df["driver_circuit_starts"] = df.groupby(["driverId", "circuitId"]).cumcount()
    df["constructor_circuit_starts"] = df.groupby(["constructorId", "circuitId"]).cumcount()

    return df.reset_index(drop=True)
