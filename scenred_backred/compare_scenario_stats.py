import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TIME_CANDIDATES = ["time", "prediction_for", "created_at", "timestamp", "datetime"]

GROUP_CONFIG = {
    "dayahead": {
        "folder": "dayahead",
        "forecast_pattern": "dayahead_forecasts_*.parquet",
    },
    "imbalance": {
        "folder": "imbalance",
        "forecast_pattern": "imbalance_forecasts_*.parquet",
    },
    "mfrr_cm_up": {
        "folder": "mfrr_cm_up",
        "forecast_pattern": "mfrr_cm_up_forecasts_*.parquet",
    },
    "mfrr_cm_down": {
        "folder": "mfrr_cm_down",
        "forecast_pattern": "mfrr_cm_down_forecasts_*.parquet",
    },
    "mfrr_eam_up": {
        "folder": "mfrr_eam_up",
        "forecast_pattern": "mfrr_eam_up_forecasts_*.parquet",
    },
    "mfrr_eam_down": {
        "folder": "mfrr_eam_down",
        "forecast_pattern": "mfrr_eam_down_forecasts_*.parquet",
    },
    "production": {
        "folder": "production",
        "forecast_pattern": "production_forecasts_*.parquet",
    },
}


@dataclass
class StatsResult:
    count: int
    mean: float
    variance: float
    skewness: float
    kurtosis: float


def parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def find_time_col(columns: Iterable[str]) -> Optional[str]:
    for cand in TIME_CANDIDATES:
        if cand in columns:
            return cand
    return None


def detect_scenario_columns(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        try:
            int(col)
        except Exception:
            continue
        out.append(str(col))
    return sorted(out, key=lambda value: int(value))


def reset_if_time_index(df: pd.DataFrame) -> pd.DataFrame:
    idx_names: list[str] = []
    if getattr(df.index, "names", None) is not None:
        idx_names = [name for name in df.index.names if name is not None]
    elif df.index.name is not None:
        idx_names = [df.index.name]

    if any(name in idx_names for name in TIME_CANDIDATES):
        return df.reset_index()
    return df


def ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)

    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)

    numeric = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    non_null_num = numeric.dropna()
    if not non_null_num.empty:
        median_abs = float(non_null_num.abs().median())
        unit = "ms" if median_abs > 1e11 else "s"
        return pd.Series(pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce"), index=series.index)

    return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)


def filter_hour_frame(
    df: pd.DataFrame,
    time_col: str,
    target_time: pd.Timestamp,
) -> pd.DataFrame:
    time_series = ensure_datetime_utc(df[time_col]).dt.floor("h")
    return df.loc[time_series == target_time].copy()


def select_single_row(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df) > 1:
        print(f"[WARNING] Multiple rows for {label}; using the first row only.")
    return df.iloc[[0]].copy()


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> StatsResult:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.size == 0:
        return StatsResult(0, np.nan, np.nan, np.nan, np.nan)

    total_weight = float(weights.sum())
    if total_weight <= 0:
        return StatsResult(int(values.size), np.nan, np.nan, np.nan, np.nan)

    mean = float(np.sum(weights * values) / total_weight)
    centered = values - mean
    variance = float(np.sum(weights * centered ** 2) / total_weight)

    if variance <= 0:
        return StatsResult(int(values.size), mean, variance, np.nan, np.nan)

    std = float(np.sqrt(variance))
    m3 = float(np.sum(weights * centered ** 3) / total_weight)
    m4 = float(np.sum(weights * centered ** 4) / total_weight)

    skewness = float(m3 / (std ** 3)) if std > 0 else np.nan
    kurtosis = float(m4 / (std ** 4)) if std > 0 else np.nan

    return StatsResult(int(values.size), mean, variance, skewness, kurtosis)


def unweighted_stats(values: np.ndarray) -> StatsResult:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return StatsResult(0, np.nan, np.nan, np.nan, np.nan)

    mean = float(np.mean(values))
    variance = float(np.var(values))

    if variance <= 0:
        return StatsResult(int(values.size), mean, variance, np.nan, np.nan)

    std = float(np.sqrt(variance))
    centered = values - mean
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))

    skewness = float(m3 / (std ** 3)) if std > 0 else np.nan
    kurtosis = float(m4 / (std ** 4)) if std > 0 else np.nan

    return StatsResult(int(values.size), mean, variance, skewness, kurtosis)


def build_output_name(forecast_path: Path) -> str:
    name = forecast_path.name
    if name.endswith(".parquet"):
        name = name.replace("_PT1H", "")
    return name


def compare_group(
    group_key: str,
    input_root: Path,
    reduced_root: Path,
    date: str,
    hour: int,
    area: Optional[str],
    park: Optional[str],
) -> dict:
    config = GROUP_CONFIG[group_key]
    folder = input_root / config["folder"]
    files = sorted(folder.glob(config["forecast_pattern"]))
    if not files:
        return {"group": group_key, "error": f"No forecast files found in {folder}"}

    forecast_path = files[0]
    original = reset_if_time_index(pd.read_parquet(forecast_path))
    time_col = find_time_col(original.columns)
    if time_col is None:
        return {"group": group_key, "error": f"No time column in {forecast_path.name}"}

    if area and "area" in original.columns:
        original = original[original["area"] == area]
    if park and "park" in original.columns:
        original = original[original["park"] == park]

    target_time = pd.to_datetime(f"{date} {hour:02d}:00:00", utc=True)
    original = filter_hour_frame(original, time_col, target_time)
    original = select_single_row(original, f"{group_key} original {date} hour {hour:02d}")

    scenario_columns = detect_scenario_columns(original.columns)
    if not scenario_columns:
        return {"group": group_key, "error": f"No scenario columns in {forecast_path.name}"}

    original_values = original.loc[:, scenario_columns].to_numpy(dtype=float).flatten()
    original_stats = unweighted_stats(original_values)

    output_name = build_output_name(forecast_path)
    reduced_path = reduced_root / output_name

    metadata_path = (
        reduced_root
        / "probabilities"
        / Path(output_name).stem
        / date
        / f"hour_{hour:02d}.json"
    )

    if not reduced_path.exists():
        return {"group": group_key, "error": f"Missing reduced parquet {reduced_path}"}
    if not metadata_path.exists():
        return {"group": group_key, "error": f"Missing metadata {metadata_path}"}

    reduced = reset_if_time_index(pd.read_parquet(reduced_path))
    if area and "area" in reduced.columns:
        reduced = reduced[reduced["area"] == area]
    if park and "park" in reduced.columns:
        reduced = reduced[reduced["park"] == park]

    reduced = filter_hour_frame(reduced, time_col, target_time)
    reduced = select_single_row(reduced, f"{group_key} reduced {date} hour {hour:02d}")

    metadata = load_metadata(metadata_path)
    output_columns = metadata.get("output_columns", [])
    probabilities = metadata.get("probabilities", [])

    if not output_columns or not probabilities:
        return {"group": group_key, "error": f"Invalid metadata {metadata_path}"}

    missing_cols = [col for col in output_columns if col not in reduced.columns]
    if missing_cols:
        return {"group": group_key, "error": f"Reduced parquet missing columns {missing_cols}"}

    reduced_values = reduced.loc[:, output_columns].to_numpy(dtype=float).flatten()
    reduced_stats = weighted_stats(reduced_values, np.asarray(probabilities))

    return {
        "group": group_key,
        "n_original": original_stats.count,
        "n_reduced": reduced_stats.count,
        "mean_original": original_stats.mean,
        "mean_reduced": reduced_stats.mean,
        "mean_delta": reduced_stats.mean - original_stats.mean,
        "variance_original": original_stats.variance,
        "variance_reduced": reduced_stats.variance,
        "variance_delta": reduced_stats.variance - original_stats.variance,
        "skew_original": original_stats.skewness,
        "skew_reduced": reduced_stats.skewness,
        "skew_delta": reduced_stats.skewness - original_stats.skewness,
        "kurtosis_original": original_stats.kurtosis,
        "kurtosis_reduced": reduced_stats.kurtosis,
        "kurtosis_delta": reduced_stats.kurtosis - original_stats.kurtosis,
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare statistics of original scenario sets vs reduced scenario sets "
            "for a specific date and hour."
        )
    )
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--hour", type=int, required=True, help="Hour of day (0-23).")
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_CONFIG.keys()),
        help="Comma-separated group keys to include.",
    )
    parser.add_argument(
        "--input-root",
        default="scenred_backred/updated_data_26",
        help="Root directory for original parquet files.",
    )
    parser.add_argument(
        "--reduced-root",
        default="scenred_backred/reduced_data_26",
        help="Root directory for reduced parquet files.",
    )
    parser.add_argument("--area", default="NO3", help="Optional area filter.")
    parser.add_argument("--park", default="roan", help="Optional park filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.hour <= 23:
        raise ValueError("--hour must be in range 0-23")

    group_keys = parse_csv_list(args.groups)
    unknown = [key for key in group_keys if key not in GROUP_CONFIG]
    if unknown:
        raise ValueError(f"Unknown groups: {', '.join(unknown)}")

    rows: list[dict] = []
    for group in group_keys:
        rows.append(
            compare_group(
                group,
                input_root=Path(args.input_root),
                reduced_root=Path(args.reduced_root),
                date=args.date,
                hour=args.hour,
                area=args.area,
                park=args.park,
            )
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("group")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda value: f"{value:0.6g}"))


if __name__ == "__main__":
    main()

'''
How to run:
python scenred_backred/compare_scenario_stats.py --date 2025-06-10 --hour 16 

optional with expanded input:
python scenred_backred/compare_scenario_stats.py --date 2025-06-10 --hour 16 \
    --groups dayahead,imbalance,mfrr_cm_up,mfrr_cm_down,mfrr_eam_up,mfrr_eam_down,production \
    --area NO3 \
    --park roan

'''