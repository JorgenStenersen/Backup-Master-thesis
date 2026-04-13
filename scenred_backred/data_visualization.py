import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

DEFAULT_PLOTS = [
    "realized_timeseries",
    "realized_distribution",
    "forecast_mean_timeseries",
    "forecast_mean_p10_p90_timeseries",
    "forecast_heatmap",
]


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def ensure_datetime_utc(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)

    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)

    # Detect if integer timestamps are in seconds or milliseconds.
    numeric = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    non_null_num = numeric.dropna()
    if not non_null_num.empty:
        median_abs = float(non_null_num.abs().median())
        unit = "ms" if median_abs > 1e11 else "s"
        return pd.Series(pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce"), index=series.index)

    return pd.Series(pd.to_datetime(series, utc=True, errors="coerce"), index=series.index)


def reset_if_time_index(df: pd.DataFrame) -> pd.DataFrame:
    idx_names = []
    if getattr(df.index, "names", None) is not None:
        idx_names = [name for name in df.index.names if name is not None]
    elif df.index.name is not None:
        idx_names = [df.index.name]

    if any(name in idx_names for name in ["time", "prediction_for", "created_at"]):
        return df.reset_index()
    return df


def read_parquet_safe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return reset_if_time_index(pd.read_parquet(path))
    except Exception as exc:
        print(f"[WARNING] Failed reading {path}: {exc}")
        return None


def find_file(folder: Path, pattern_or_name: str) -> Optional[Path]:
    if "*" in pattern_or_name:
        files = sorted(folder.glob(pattern_or_name))
        return files[0] if files else None

    candidate = folder / pattern_or_name
    return candidate if candidate.exists() else None


def detect_horizon_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        try:
            int(col)
            cols.append(col)
        except Exception:
            continue
    return sorted(cols, key=lambda value: int(value))


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
        if col in df.columns:
            return col

    excluded = {"time", "prediction_for", "created_at", "area", "park", "time_dt", "prediction_for_dt"}
    numeric_candidates = [
        col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_candidates[0] if numeric_candidates else None


def format_market_name(group_label: str) -> str:
    return group_label.replace("mFrr", "mFRR").replace("Mfrr", "mFRR")


def format_realized_label(value_col: str, group_label: str) -> str:
    group_name = format_market_name(group_label)
    group_key = group_label.strip().lower()

    if group_key.startswith("dayahead"):
        return "Price (€/MWh)"

    if group_key.startswith("imbalance"):
        return "Price (€/MWh)"

    if group_key.startswith("production"):
        return "Production (MW)"

    if "cm" in group_key:
        return "Price (€/MWh)"

    if "eam" in group_key:
        return "Price (€/MWh)"

    label_map = {
        "dayahead_price": "Price (€/MWh)",
        "imbalance_price": "Price (€/MWh)",
        "mfrr_price": f"{group_name} price (€/MWh)",
        "production": "Production (MW)",
        "price": f"{group_name} price (€/MWh)",
        "value": "Value",
    }
    if value_col in label_map:
        return label_map[value_col]
    return value_col.replace("_", " ").strip().capitalize()


def apply_common_filters(
    df: pd.DataFrame,
    area: Optional[str],
    park: Optional[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df.copy()

    if area and "area" in out.columns:
        out = out[out["area"] == area]

    if park and "park" in out.columns:
        out = out[out["park"] == park]

    if "time" in out.columns:
        out["time_dt"] = ensure_datetime_utc(out["time"])
        if start is not None:
            out = out[out["time_dt"] >= start]
        if end is not None:
            out = out[out["time_dt"] <= end]

    if "prediction_for" in out.columns:
        out["prediction_for_dt"] = ensure_datetime_utc(out["prediction_for"])
        if start is not None:
            out = out[out["prediction_for_dt"] >= start]
        if end is not None:
            out = out[out["prediction_for_dt"] <= end]

    if "created_at" in out.columns:
        out["created_at_dt"] = ensure_datetime_utc(out["created_at"])

    return out


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def forecast_axis_labels(group_label: str) -> Tuple[str, str]:
    if group_label.strip().lower() == "production":
        return "Mean Production (MW)", "Production (MW)"
    return "Mean price (€/MWh)", "Price (€/MWh)"


def save_or_show(fig, save_path: Optional[Path]) -> None:
    if save_path is None:
        plt.show()
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_realized_timeseries(df_realized: Optional[pd.DataFrame], group_label: str, save_path: Optional[Path]) -> None:
    if df_realized is None or df_realized.empty:
        print(f"[INFO] No realized data available for {group_label}")
        return

    if "time_dt" not in df_realized.columns:
        print(f"[INFO] Skipping realized_timeseries for {group_label}: missing time information")
        return

    value_col = detect_realized_value_column(df_realized)
    if value_col is None:
        print(f"[INFO] Skipping realized_timeseries for {group_label}: missing numeric value column")
        return

    data = df_realized[["time_dt", str(value_col)]].dropna().sort_values(by="time_dt")
    if data.empty:
        print(f"[INFO] No rows to plot for {group_label} realized_timeseries")
        return

    fig, ax = plt.subplots()
    ax.plot(
        data["time_dt"],
        data[value_col],
        lw=1.5,
        color="#1f77b4",
        label="Realized production" if group_label.strip().lower().startswith("production") else "Realized price",
    )

    if len(data) >= 24:
        rolling = data[value_col].rolling(24, min_periods=1).mean()
        ax.plot(data["time_dt"], rolling, lw=2.0, color="#d62728", label="24h rolling mean")

    group_key = group_label.strip().lower()
    if group_key.startswith("production"):
        title_text = "Realized production time series"
    else:
        title_text = "Realized price time series"
    ax.set_title(f"{group_label} - {title_text}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(format_realized_label(value_col, group_label))
    ax.legend()
    fig.autofmt_xdate()
    save_or_show(fig, save_path)


def plot_realized_distribution(
    df_realized: Optional[pd.DataFrame], group_label: str, save_path: Optional[Path]
) -> None:
    if df_realized is None or df_realized.empty:
        return

    value_col = detect_realized_value_column(df_realized)
    if value_col is None:
        print(f"[INFO] Skipping realized_distribution for {group_label}: missing numeric value column")
        return

    values = df_realized[value_col].dropna()
    if values.empty:
        return

    fig, ax = plt.subplots()
    sns.histplot(values, bins=40, kde=True, color="#2ca02c", ax=ax)
    ax.set_title(f"{group_label} - Realized Distribution")
    ax.set_xlabel(format_realized_label(value_col, group_label))
    ax.set_ylabel("Count")
    save_or_show(fig, save_path)


def plot_forecast_mean_p10_p90_timeseries(
    df_forecast: Optional[pd.DataFrame], group_label: str, save_path: Optional[Path]
) -> None:
    if df_forecast is None or df_forecast.empty:
        print(f"[INFO] No forecast data available for {group_label}")
        return

    horizon_cols = detect_horizon_columns(df_forecast)
    if not horizon_cols:
        print(f"[INFO] Skipping forecast_mean_p10_p90_timeseries for {group_label}: no horizon columns found")
        return

    time_col = None
    if "prediction_for_dt" in df_forecast.columns:
        time_col = "prediction_for_dt"
    elif "time_dt" in df_forecast.columns:
        time_col = "time_dt"
    elif "created_at_dt" in df_forecast.columns:
        time_col = "created_at_dt"

    if time_col is None:
        print(f"[INFO] Skipping forecast_mean_p10_p90_timeseries for {group_label}: missing time information")
        return

    numeric_horizons = df_forecast[horizon_cols].apply(pd.to_numeric, errors="coerce")
    q10 = numeric_horizons.quantile(0.10, axis=1)
    mean_series = numeric_horizons.mean(axis=1)
    q90 = numeric_horizons.quantile(0.90, axis=1)

    data = pd.DataFrame(
        {
            "time_dt": ensure_datetime_utc(df_forecast[time_col]),
            "q10": q10,
            "mean": mean_series,
            "q90": q90,
        }
    ).dropna()

    if data.empty:
        print(f"[INFO] No rows to plot for {group_label} forecast_mean_p10_p90_timeseries")
        return

    data = (
        data.groupby("time_dt", as_index=False)[["q10", "mean", "q90"]]
        .mean()
        .sort_values("time_dt")
    )

    fig, ax = plt.subplots()
    _, value_ylabel = forecast_axis_labels(group_label)
    ax.fill_between(data["time_dt"], data["q10"], data["q90"], alpha=0.25, color="#1f77b4", label="P10-P90")
    ax.plot(data["time_dt"], data["mean"], color="#d62728", lw=1.8, label="Mean")
    ax.set_title(f"{group_label} - Forecast Mean with P10-P90 Over Time")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(value_ylabel)
    ax.legend()
    fig.autofmt_xdate()
    save_or_show(fig, save_path)


def plot_forecast_mean_timeseries(
    df_forecast: Optional[pd.DataFrame], group_label: str, save_path: Optional[Path]
) -> None:
    if df_forecast is None or df_forecast.empty:
        print(f"[INFO] No forecast data available for {group_label}")
        return

    horizon_cols = detect_horizon_columns(df_forecast)
    if not horizon_cols:
        print(f"[INFO] Skipping forecast_mean_timeseries for {group_label}: no horizon columns found")
        return

    time_col = None
    if "prediction_for_dt" in df_forecast.columns:
        time_col = "prediction_for_dt"
    elif "time_dt" in df_forecast.columns:
        time_col = "time_dt"
    elif "created_at_dt" in df_forecast.columns:
        time_col = "created_at_dt"

    if time_col is None:
        print(f"[INFO] Skipping forecast_mean_timeseries for {group_label}: missing time information")
        return

    # Average across all horizon/member columns for each time step.
    numeric_horizons = df_forecast[horizon_cols].apply(pd.to_numeric, errors="coerce")
    mean_series = numeric_horizons.mean(axis=1)

    data = pd.DataFrame(
        {
            "time_dt": ensure_datetime_utc(df_forecast[time_col]),
            "forecast_mean": mean_series,
        }
    ).dropna()

    if data.empty:
        print(f"[INFO] No rows to plot for {group_label} forecast_mean_timeseries")
        return

    # If duplicate timestamps exist, aggregate them to one hourly value.
    data = data.groupby("time_dt", as_index=False)["forecast_mean"].mean().sort_values("time_dt")

    fig, ax = plt.subplots()
    mean_ylabel, _ = forecast_axis_labels(group_label)
    ax.plot(data["time_dt"], data["forecast_mean"], lw=1.5, color="#1f77b4", label="Forecast mean")

    if len(data) >= 24:
        rolling = data["forecast_mean"].rolling(24, min_periods=1).mean()
        ax.plot(data["time_dt"], rolling, lw=2.0, color="#d62728", label="24h rolling mean")

    ax.set_title(f"{group_label} - Forecast Mean Time Series")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(mean_ylabel)
    ax.legend()
    fig.autofmt_xdate()
    save_or_show(fig, save_path)


def plot_forecast_heatmap(
    df_forecast: Optional[pd.DataFrame], group_label: str, max_forecasts: int, save_path: Optional[Path]
) -> None:
    if df_forecast is None or df_forecast.empty:
        return

    horizon_cols = detect_horizon_columns(df_forecast)
    if not horizon_cols:
        return

    data = df_forecast.copy()
    if "created_at_dt" in data.columns:
        data = data.sort_values("created_at_dt")
    elif "prediction_for_dt" in data.columns:
        data = data.sort_values("prediction_for_dt")

    if max_forecasts > 0 and len(data) > max_forecasts:
        data = data.tail(max_forecasts)

    matrix = data[horizon_cols].copy()
    if not isinstance(matrix, pd.DataFrame):
        return
    matrix.columns = [int(col) for col in matrix.columns]
    matrix = matrix.sort_index(axis=1)

    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(matrix, cmap="viridis", cbar=True, ax=ax)
    ax.set_title(f"{group_label} - Forecast Heatmap (rows=vintages, cols=horizons)")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Forecast vintage index")
    save_or_show(fig, save_path)


def load_group_data(
    base_dir: Path,
    group_key: str,
    area: Optional[str],
    park: Optional[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cfg = GROUP_CONFIG[group_key]
    folder = base_dir / cfg["folder"]

    forecast_file = find_file(folder, cfg["forecast_pattern"])
    realized_file = find_file(folder, cfg["realized_pattern"])

    df_forecast = read_parquet_safe(forecast_file)
    df_realized = read_parquet_safe(realized_file)

    if df_forecast is not None:
        df_forecast = apply_common_filters(df_forecast, area=area, park=park, start=start, end=end)

    if df_realized is not None:
        df_realized = apply_common_filters(df_realized, area=area, park=park, start=start, end=end)

    return df_forecast, df_realized


def run_group(
    base_dir: Path,
    group_key: str,
    plots: List[str],
    area: Optional[str],
    park: Optional[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_forecasts: int,
    save_dir: Optional[Path],
) -> None:
    cfg = GROUP_CONFIG[group_key]
    label = cfg["label"]

    print(f"[INFO] Processing group: {group_key}")
    df_forecast, df_realized = load_group_data(base_dir, group_key, area, park, start, end)

    if df_forecast is None and df_realized is None:
        print(f"[WARNING] No data found for {group_key}")
        return

    def output_file(name: str) -> Optional[Path]:
        if save_dir is None:
            return None
        return save_dir / group_key / f"{name}.png"

    if "realized_timeseries" in plots:
        plot_realized_timeseries(df_realized, label, output_file("realized_timeseries"))

    if "realized_distribution" in plots:
        plot_realized_distribution(df_realized, label, output_file("realized_distribution"))

    if "forecast_mean_timeseries" in plots:
        plot_forecast_mean_timeseries(df_forecast, label, output_file("forecast_mean_timeseries"))

    if "forecast_mean_p10_p90_timeseries" in plots:
        plot_forecast_mean_p10_p90_timeseries(
            df_forecast, label, output_file("forecast_mean_p10_p90_timeseries")
        )

    if "forecast_heatmap" in plots:
        plot_forecast_heatmap(df_forecast, label, max_forecasts, output_file("forecast_heatmap"))


def parse_time(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(parsed, pd.Timestamp):
        return parsed
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize parquet market data from updated_data_26")
    parser.add_argument("--base-dir", default="updated_data_26", help="Root directory containing market subfolders")
    parser.add_argument(
        "--groups",
        default="dayahead,imbalance,mfrr_cm_up,mfrr_cm_down,mfrr_eam_up,mfrr_eam_down,production",
        help="Comma-separated groups to visualize",
    )
    parser.add_argument(
        "--plots",
        default=",".join(DEFAULT_PLOTS),
        help="Comma-separated plot types: realized_timeseries,realized_distribution,forecast_mean_timeseries,forecast_mean_p10_p90_timeseries,forecast_heatmap",
    )
    parser.add_argument("--area", default=None, help="Optional area filter")
    parser.add_argument("--park", default=None, help="Optional park filter")
    parser.add_argument("--start", default=None, help="Optional start UTC timestamp")
    parser.add_argument("--end", default=None, help="Optional end UTC timestamp")
    parser.add_argument(
        "--max-forecasts",
        type=int,
        default=250,
        help="Maximum number of forecast rows for heatmap/profile plots",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, saves .png files in this directory instead of opening plot windows",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None
    groups = parse_csv_list(args.groups)
    plots = parse_csv_list(args.plots)
    start = parse_time(args.start)
    end = parse_time(args.end)

    valid_groups = [group for group in groups if group in GROUP_CONFIG]
    invalid_groups = [group for group in groups if group not in GROUP_CONFIG]

    if invalid_groups:
        print(f"[WARNING] Unknown groups ignored: {invalid_groups}")

    if not valid_groups:
        raise ValueError("No valid groups selected. Use --groups with keys in GROUP_CONFIG.")

    setup_plot_style()

    for group_key in valid_groups:
        run_group(
            base_dir=base_dir,
            group_key=group_key,
            plots=plots,
            area=args.area,
            park=args.park,
            start=start,
            end=end,
            max_forecasts=args.max_forecasts,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
