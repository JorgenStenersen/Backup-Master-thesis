import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


GROUP_CONFIG = {
    "dayahead": {
        "folder": "dayahead",
        "forecast_pattern": "dayahead_forecasts_*.parquet",
        "realized_pattern": "dayahead_prices_*.parquet",
        "label": "Dayahead",
    },
    "imbalance": {
        "folder": "imbalance",
        "forecast_pattern": "imbalance_forecasts_*.parquet",
        "realized_pattern": "imbalance_prices_*.parquet",
        "label": "Imbalance",
    },
    "mfrr_cm_up": {
        "folder": "mfrr_cm_up",
        "forecast_pattern": "mfrr_cm_up_forecasts_*.parquet",
        "realized_pattern": "mfrr_cm_up_prices_*.parquet",
        "label": "mFRR CM Up",
    },
    "mfrr_cm_down": {
        "folder": "mfrr_cm_down",
        "forecast_pattern": "mfrr_cm_down_forecasts_*.parquet",
        "realized_pattern": "mfrr_cm_down_prices_*.parquet",
        "label": "mFRR CM Down",
    },
    "mfrr_eam_up": {
        "folder": "mfrr_eam_up",
        "forecast_pattern": "mfrr_eam_up_forecasts_*.parquet",
        "realized_pattern": "mfrr_eam_up_prices_*.parquet",
        "label": "mFRR EAM Up",
    },
    "mfrr_eam_down": {
        "folder": "mfrr_eam_down",
        "forecast_pattern": "mfrr_eam_down_forecasts_*.parquet",
        "realized_pattern": "mfrr_eam_down_prices_*.parquet",
        "label": "mFRR EAM Down",
    },
    "production": {
        "folder": "production",
        "forecast_pattern": "production_forecasts_*.parquet",
        "realized_pattern": "production.parquet",
        "label": "Production",
    },
}

TIME_LIKE_COLUMNS = {
    "time",
    "prediction_for",
    "created_at",
    "timestamp",
    "datetime",
    "time_dt",
    "prediction_for_dt",
    "created_at_dt",
    "area",
    "park",
}


@dataclass
class BoundsSummary:
    group_key: str
    group_label: str
    dataset_type: str
    file: str
    value_source: str
    observations: int
    global_min: Optional[float]
    global_max: Optional[float]
    p10: Optional[float]
    p90: Optional[float]
    global_span: Optional[float]
    p10_p90_span: Optional[float]
    error: Optional[str]


def parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve global and P10/P90 bounds per market from parquet files "
            "(realized and/or forecast)."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="updated_data_26",
        help="Root directory containing market subfolders.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_CONFIG.keys()),
        help="Comma-separated groups to inspect.",
    )
    parser.add_argument(
        "--include-realized",
        action="store_true",
        help="Include realized files in the output.",
    )
    parser.add_argument(
        "--include-forecasts",
        action="store_true",
        help="Include forecast files in the output.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def find_file(folder: Path, pattern_or_name: str) -> Optional[Path]:
    if "*" in pattern_or_name:
        files = sorted(folder.glob(pattern_or_name))
        return files[0] if files else None

    candidate = folder / pattern_or_name
    return candidate if candidate.exists() else None


def detect_horizon_columns(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        try:
            int(col)
            out.append(col)
        except Exception:
            continue
    return sorted(out, key=lambda x: int(x))


def detect_realized_value_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "dayahead_price",
        "imbalance_price",
        "mfrr_price",
        "production",
        "price",
        "value",
    ]
    for col in preferred:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col

    numeric_candidates = [
        col
        for col in df.columns
        if col not in TIME_LIKE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_candidates[0] if numeric_candidates else None


def to_float_or_none(value: float) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def compute_bounds(values: pd.Series) -> dict[str, Optional[float] | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "observations": 0,
            "global_min": None,
            "global_max": None,
            "p10": None,
            "p90": None,
            "global_span": None,
            "p10_p90_span": None,
        }

    global_min = float(clean.min())
    global_max = float(clean.max())
    p10 = float(clean.quantile(0.10))
    p90 = float(clean.quantile(0.90))

    return {
        "observations": int(clean.shape[0]),
        "global_min": global_min,
        "global_max": global_max,
        "p10": p10,
        "p90": p90,
        "global_span": float(global_max - global_min),
        "p10_p90_span": float(p90 - p10),
    }


def summarize_realized(group_key: str, group_label: str, file_path: Path) -> BoundsSummary:
    try:
        df = pd.read_parquet(file_path)
        value_col = detect_realized_value_column(df)
        if value_col is None:
            return BoundsSummary(
                group_key=group_key,
                group_label=group_label,
                dataset_type="realized",
                file=str(file_path),
                value_source="none",
                observations=0,
                global_min=None,
                global_max=None,
                p10=None,
                p90=None,
                global_span=None,
                p10_p90_span=None,
                error="No numeric realized value column found.",
            )

        bounds = compute_bounds(df[value_col])
        return BoundsSummary(
            group_key=group_key,
            group_label=group_label,
            dataset_type="realized",
            file=str(file_path),
            value_source=value_col,
            observations=int(bounds["observations"]),
            global_min=to_float_or_none(bounds["global_min"]),
            global_max=to_float_or_none(bounds["global_max"]),
            p10=to_float_or_none(bounds["p10"]),
            p90=to_float_or_none(bounds["p90"]),
            global_span=to_float_or_none(bounds["global_span"]),
            p10_p90_span=to_float_or_none(bounds["p10_p90_span"]),
            error=None,
        )
    except Exception as exc:
        return BoundsSummary(
            group_key=group_key,
            group_label=group_label,
            dataset_type="realized",
            file=str(file_path),
            value_source="unknown",
            observations=0,
            global_min=None,
            global_max=None,
            p10=None,
            p90=None,
            global_span=None,
            p10_p90_span=None,
            error=str(exc),
        )


def summarize_forecast(group_key: str, group_label: str, file_path: Path) -> BoundsSummary:
    try:
        df = pd.read_parquet(file_path)
        horizon_cols = detect_horizon_columns(df.columns)
        if not horizon_cols:
            return BoundsSummary(
                group_key=group_key,
                group_label=group_label,
                dataset_type="forecast",
                file=str(file_path),
                value_source="none",
                observations=0,
                global_min=None,
                global_max=None,
                p10=None,
                p90=None,
                global_span=None,
                p10_p90_span=None,
                error="No horizon columns found.",
            )

        # Flatten all horizon values to one vector per market so bounds represent
        # the full forecast value space for that market.
        values = pd.Series(df[horizon_cols].to_numpy().reshape(-1))
        bounds = compute_bounds(values)

        return BoundsSummary(
            group_key=group_key,
            group_label=group_label,
            dataset_type="forecast",
            file=str(file_path),
            value_source="all_horizons_flattened",
            observations=int(bounds["observations"]),
            global_min=to_float_or_none(bounds["global_min"]),
            global_max=to_float_or_none(bounds["global_max"]),
            p10=to_float_or_none(bounds["p10"]),
            p90=to_float_or_none(bounds["p90"]),
            global_span=to_float_or_none(bounds["global_span"]),
            p10_p90_span=to_float_or_none(bounds["p10_p90_span"]),
            error=None,
        )
    except Exception as exc:
        return BoundsSummary(
            group_key=group_key,
            group_label=group_label,
            dataset_type="forecast",
            file=str(file_path),
            value_source="unknown",
            observations=0,
            global_min=None,
            global_max=None,
            p10=None,
            p90=None,
            global_span=None,
            p10_p90_span=None,
            error=str(exc),
        )


def main() -> None:
    args = parse_args()

    include_realized = args.include_realized
    include_forecasts = args.include_forecasts

    # Default behavior: include both when neither flag is explicitly set.
    if not include_realized and not include_forecasts:
        include_realized = True
        include_forecasts = True

    base_dir = Path(args.base_dir)
    selected_groups = parse_csv_list(args.groups)

    valid_groups = [group for group in selected_groups if group in GROUP_CONFIG]
    invalid_groups = [group for group in selected_groups if group not in GROUP_CONFIG]

    if invalid_groups:
        print(f"[WARNING] Unknown groups ignored: {invalid_groups}")

    if not valid_groups:
        raise ValueError("No valid groups selected. Use --groups with keys from GROUP_CONFIG.")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    summaries: list[BoundsSummary] = []

    for group_key in valid_groups:
        cfg = GROUP_CONFIG[group_key]
        group_label = cfg["label"]
        folder = base_dir / cfg["folder"]

        if not folder.exists():
            summaries.append(
                BoundsSummary(
                    group_key=group_key,
                    group_label=group_label,
                    dataset_type="folder",
                    file=str(folder),
                    value_source="none",
                    observations=0,
                    global_min=None,
                    global_max=None,
                    p10=None,
                    p90=None,
                    global_span=None,
                    p10_p90_span=None,
                    error="Market folder does not exist.",
                )
            )
            continue

        if include_realized:
            realized_file = find_file(folder, cfg["realized_pattern"])
            if realized_file is None:
                summaries.append(
                    BoundsSummary(
                        group_key=group_key,
                        group_label=group_label,
                        dataset_type="realized",
                        file=str(folder / cfg["realized_pattern"]),
                        value_source="none",
                        observations=0,
                        global_min=None,
                        global_max=None,
                        p10=None,
                        p90=None,
                        global_span=None,
                        p10_p90_span=None,
                        error="Realized file not found.",
                    )
                )
            else:
                summaries.append(summarize_realized(group_key, group_label, realized_file))

        if include_forecasts:
            forecast_file = find_file(folder, cfg["forecast_pattern"])
            if forecast_file is None:
                summaries.append(
                    BoundsSummary(
                        group_key=group_key,
                        group_label=group_label,
                        dataset_type="forecast",
                        file=str(folder / cfg["forecast_pattern"]),
                        value_source="none",
                        observations=0,
                        global_min=None,
                        global_max=None,
                        p10=None,
                        p90=None,
                        global_span=None,
                        p10_p90_span=None,
                        error="Forecast file not found.",
                    )
                )
            else:
                summaries.append(summarize_forecast(group_key, group_label, forecast_file))

    df = pd.DataFrame(asdict(row) for row in summaries)

    display_cols = [
        "group_key",
        "dataset_type",
        "observations",
        "global_min",
        "global_max",
        "p10",
        "p90",
        "global_span",
        "p10_p90_span",
        "value_source",
        "error",
    ]

    print("\nBounds summary:\n")
    print(df[display_cols].to_string(index=False))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nWrote CSV summary to {output_csv}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as file:
            json.dump([asdict(row) for row in summaries], file, indent=2)
        print(f"Wrote JSON summary to {output_json}")


if __name__ == "__main__":
    main()
