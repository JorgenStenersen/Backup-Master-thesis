import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq


TIME_CANDIDATES = ["time", "prediction_for", "created_at", "timestamp", "datetime"]


@dataclass
class FileSummary:
    file: str
    rows: Optional[int]
    cols: Optional[int]
    row_groups: Optional[int]
    size_mb: float
    rows_eq_8760: Optional[bool]
    kind: str
    horizon_count: int
    horizon_min: Optional[int]
    horizon_max: Optional[int]
    time_col: Optional[str]
    min_time_utc: Optional[str]
    max_time_utc: Optional[str]
    unique_hours: Optional[int]
    expected_hours: Optional[int]
    missing_hours: Optional[int]
    duplicate_timestamps: Optional[int]
    error: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect parquet files in a structured, efficient way."
    )
    parser.add_argument(
        "--base-dir",
        default="updated_data_26",
        help="Root directory to scan recursively for parquet files.",
    )
    parser.add_argument(
        "--glob",
        default="*.parquet",
        help="File glob pattern used with recursive search (default: *.parquet).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path for the summary table.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for the summary list.",
    )
    parser.add_argument(
        "--show-columns",
        action="store_true",
        help="Also print schema columns per file.",
    )
    return parser.parse_args()


def safe_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    return ts.isoformat()


def find_time_col(columns: list[str]) -> Optional[str]:
    for cand in TIME_CANDIDATES:
        if cand in columns:
            return cand
    return None


def detect_horizons(columns: list[str]) -> list[int]:
    out: list[int] = []
    for col in columns:
        try:
            out.append(int(col))
        except Exception:
            continue
    return sorted(out)


def summarize_time_dimension(file_path: Path, time_col: str) -> dict[str, Optional[int | str]]:
    # Read only the time column to keep memory and IO low.
    frame = pd.read_parquet(file_path, columns=[time_col])
    if time_col in frame.columns:
        ts = frame[time_col]
    elif frame.index.name == time_col:
        ts = pd.Series(frame.index, index=frame.index)
    else:
        # Fall back to full read for edge cases where parquet metadata stores
        # pandas index/column information in a non-standard way.
        frame = pd.read_parquet(file_path)
        if time_col in frame.columns:
            ts = frame[time_col]
        elif frame.index.name == time_col:
            ts = pd.Series(frame.index, index=frame.index)
        else:
            raise KeyError(time_col)
    ts = pd.to_datetime(ts, utc=True, errors="coerce").dropna()

    if ts.empty:
        return {
            "min_time_utc": None,
            "max_time_utc": None,
            "unique_hours": 0,
            "expected_hours": 0,
            "missing_hours": 0,
            "duplicate_timestamps": 0,
        }

    ts_hours = ts.dt.floor("h")
    unique_hours = int(ts_hours.nunique())
    min_time = ts.min()
    max_time = ts.max()
    expected = int((max_time - min_time).total_seconds() // 3600) + 1
    missing = max(expected - unique_hours, 0)
    duplicates = int(ts.duplicated().sum())

    return {
        "min_time_utc": safe_iso(min_time),
        "max_time_utc": safe_iso(max_time),
        "unique_hours": unique_hours,
        "expected_hours": expected,
        "missing_hours": missing,
        "duplicate_timestamps": duplicates,
    }


def summarize_one(file_path: Path, show_columns: bool) -> FileSummary:
    size_mb = round(file_path.stat().st_size / (1024 * 1024), 3)

    try:
        parquet_file = pq.ParquetFile(file_path)
        meta = parquet_file.metadata
        schema = parquet_file.schema_arrow
        columns = list(schema.names)

        horizons = detect_horizons(columns)
        time_col = find_time_col(columns)
        kind = "forecast" if horizons else "timeseries"

        time_summary: dict[str, Any] = {
            "min_time_utc": None,
            "max_time_utc": None,
            "unique_hours": None,
            "expected_hours": None,
            "missing_hours": None,
            "duplicate_timestamps": None,
        }
        if time_col is not None:
            try:
                time_summary = summarize_time_dimension(file_path, time_col)
            except Exception as exc:
                time_summary = {
                    "min_time_utc": None,
                    "max_time_utc": None,
                    "unique_hours": None,
                    "expected_hours": None,
                    "missing_hours": None,
                    "duplicate_timestamps": None,
                }
                print(f"[WARNING] Could not summarize time for {file_path}: {exc}")

        if show_columns:
            print(f"\nColumns in {file_path}:")
            print(", ".join(columns))

        rows = int(meta.num_rows)
        return FileSummary(
            file=str(file_path),
            rows=rows,
            cols=int(meta.num_columns),
            row_groups=int(meta.num_row_groups),
            size_mb=size_mb,
            rows_eq_8760=(rows == 8760),
            kind=kind,
            horizon_count=len(horizons),
            horizon_min=(horizons[0] if horizons else None),
            horizon_max=(horizons[-1] if horizons else None),
            time_col=time_col,
            min_time_utc=time_summary["min_time_utc"],
            max_time_utc=time_summary["max_time_utc"],
            unique_hours=time_summary["unique_hours"],
            expected_hours=time_summary["expected_hours"],
            missing_hours=time_summary["missing_hours"],
            duplicate_timestamps=time_summary["duplicate_timestamps"],
            error=None,
        )
    except Exception as exc:
        return FileSummary(
            file=str(file_path),
            rows=None,
            cols=None,
            row_groups=None,
            size_mb=size_mb,
            rows_eq_8760=None,
            kind="unknown",
            horizon_count=0,
            horizon_min=None,
            horizon_max=None,
            time_col=None,
            min_time_utc=None,
            max_time_utc=None,
            unique_hours=None,
            expected_hours=None,
            missing_hours=None,
            duplicate_timestamps=None,
            error=str(exc),
        )


def discover_files(base_dir: Path, glob_pattern: str) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob(glob_pattern))


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    files = discover_files(base_dir, args.glob)

    if not files:
        print(f"No parquet files found in {base_dir} using pattern {args.glob}")
        return

    print(f"Found {len(files)} parquet files in {base_dir}\n")
    summaries = [summarize_one(file_path, show_columns=args.show_columns) for file_path in files]

    df = pd.DataFrame(asdict(row) for row in summaries)
    display_cols = [
        "file",
        "rows",
        "rows_eq_8760",
        "kind",
        "time_col",
        "unique_hours",
        "missing_hours",
        "horizon_count",
        "size_mb",
        "error",
    ]
    print(df[display_cols].to_string(index=False))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nWrote CSV summary to {output_csv}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump([asdict(row) for row in summaries], f, indent=2)
        print(f"Wrote JSON summary to {output_json}")


if __name__ == "__main__":
    main()


'''
Usage examples
c:/Master/scenred_Jonas_Engels/.venv/Scripts/python.exe parquet_inspector.py --base-dir updated_data_26

c:/Master/scenred_Jonas_Engels/.venv/Scripts/python.exe parquet_inspector.py --base-dir updated_data_26 --output-csv reports/parquet_summary.csv --output-json reports/parquet_summary.json

c:/Master/scenred_Jonas_Engels/.venv/Scripts/python.exe parquet_inspector.py --base-dir updated_data_26 --show-columns
'''