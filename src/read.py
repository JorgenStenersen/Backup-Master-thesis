import pandas as pd
import numpy as np
import pyarrow
import fastparquet
import os
from datetime import datetime
from src.utils import select_scenarios


def load_parameters_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer seks lister:
    CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed
    """
    df = pd.read_csv(path)

    CM_up      = df["CM_up"].tolist()
    CM_down    = df["CM_down"].tolist()
    DA         = df["DA"].tolist()
    EAM_up     = df["EAM_up"].tolist()
    EAM_down   = df["EAM_down"].tolist()
    wind_speed = df["wind_speed"].tolist()

    return CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed



def load_expected_values_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer forventede verdier:
    P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean
    """
    df = pd.read_csv(path)

    # Forventede (gjennomsnittlige) priser og vind
    P_CM_up    = df["CM_up"].mean()
    P_CM_down  = df["CM_down"].mean()
    P_DA       = df["DA"].mean()
    P_EAM_up   = df["EAM_up"].mean()
    P_EAM_down = df["EAM_down"].mean()
    Q_mean     = df["wind_speed"].mean()   # tilgjengelig produksjonskapasitet

    return P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean

def _reset_if_time_in_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'time' or 'prediction_for' is in the index (or a MultiIndex),
    reset the index so they become normal columns.
    """
    df = df.copy()

    # Collect index names (works for Index and MultiIndex)
    index_names = []
    if getattr(df.index, "names", None) is not None:
        index_names = [n for n in df.index.names if n is not None]
    elif df.index.name is not None:
        index_names = [df.index.name]

    if any(name in index_names for name in ["time", "prediction_for"]):
        df = df.reset_index()

    return df


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Convert int timestamps (seconds or ms since epoch) to UTC datetime.
    Tries to auto-detect unit based on magnitude.
    """
    s = series.dropna()
    if s.empty:
        # Fallback, assume ms
        return pd.to_datetime(series, unit="ms", utc=True)

    median_abs = float(s.astype("int64").abs().median())
    # crude threshold: >1e11 ~ ms since 1970; <1e11 ~ seconds
    unit = "ms" if median_abs > 1e11 else "s"
    return pd.to_datetime(series, unit=unit, utc=True)


def load_market_data(
    time_str: str,
    raw_path: str = "data",
    area: str = "NO3",
    park: str = "roan",
):
    """
    Load all relevant market data for a given time (string, UTC),
    and return a dict with:
      - forecast files: lists of 50 values (columns '0'..'49')
        for the FIRST forecast (by created_at) with prediction_for == time
      - realized price/production files: the value at that exact time
    """

    target_time = pd.to_datetime(time_str, utc=True)
    results = {}

    file_mapping = {
        "dayahead_forecasts.parquet": "dayahead_forecasts",
        "dayahead_prices.parquet": "dayahead_prices",
        "imbalance_forecasts.parquet": "imbalance_forecasts",
        "imbalance_prices.parquet": "imbalance_prices",
        "mfrr_cm_down_forecasts.parquet": "mfrr_cm_down_forecasts",
        "mfrr_cm_down_prices.parquet": "mfrr_cm_down_prices",
        "mfrr_cm_up_forecasts.parquet": "mfrr_cm_up_forecasts",
        "mfrr_cm_up_prices.parquet": "mfrr_cm_up_prices",
        "mfrr_eam_down_forecasts.parquet": "mfrr_eam_down_forecasts",
        "mfrr_eam_down_prices.parquet": "mfrr_eam_down_prices",
        "mfrr_eam_up_forecasts.parquet": "mfrr_eam_up_forecasts",
        "mfrr_eam_up_prices.parquet": "mfrr_eam_up_prices",
        "production_forecasts.parquet": "production_forecasts",
        "production.parquet": "production",
    }

    horizon_cols = [str(i) for i in range(50)]

    for filename, var_name in file_mapping.items():
        file_path = os.path.join(raw_path, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            results[var_name] = None
            continue

        df = pd.read_parquet(file_path)
        df = _reset_if_time_in_index(df)  # <--- important fix

        # --- Forecast-style files ---
        if "prediction_for" in df.columns:
            df = df.copy()
            df["prediction_for_dt"] = _to_utc_datetime(df["prediction_for"])
            mask = df["prediction_for_dt"] == target_time

            if "area" in df.columns:
                mask &= df["area"] == area
            if "park" in df.columns:
                mask &= df["park"] == park

            df_match = df.loc[mask]

            if df_match.empty:
                # Small debug helper: show min/max available prediction_for
                pf_sample = df["prediction_for_dt"].sort_values().dropna()
                if not pf_sample.empty:
                    print(
                        f"⚠️ No forecast rows in {filename} for {target_time}. "
                        f"Available range: {pf_sample.iloc[0]} – {pf_sample.iloc[-1]}"
                    )
                else:
                    print(
                        f"⚠️ No forecast rows in {filename} for {target_time} "
                        f"and prediction_for_dt is empty/NaN."
                    )
                results[var_name] = None
                continue

            # FIRST forecast: earliest created_at
            if "created_at" in df_match.columns:
                df_match = df_match.sort_values("created_at").head(1)
                print(f"Using forecast from {var_name}, created_at: {df_match['created_at'].iloc[0]}")
            else:
                df_match = df_match.head(1)
                print("No created_at column; using first row.")

            cols_present = [c for c in horizon_cols if c in df_match.columns]
            if not cols_present:
                print(f"⚠️ No 0..49 columns in {filename}")
                results[var_name] = None
                continue

            row = df_match.iloc[0]
            results[var_name] = row[cols_present].astype(float).tolist()

        # --- Realized-style files (prices & production) ---
        elif "time" in df.columns:
            df = df.copy()
            df["time_dt"] = _to_utc_datetime(df["time"])
            mask = df["time_dt"] == target_time

            if "area" in df.columns:
                mask &= df["area"] == area
            if "park" in df.columns:
                mask &= df["park"] == park

            df_match = df.loc[mask]

            if df_match.empty:
                t_sample = df["time_dt"].sort_values().dropna()
                if not t_sample.empty:
                    print(
                        f"⚠️ No realized rows in {filename} for {target_time}. "
                        f"Available range: {t_sample.iloc[0]} – {t_sample.iloc[-1]}"
                    )
                else:
                    print(
                        f"⚠️ No realized rows in {filename} for {target_time} "
                        f"and time_dt is empty/NaN."
                    )
                results[var_name] = None
                continue

            value_cols = [
                c
                for c in [
                    "dayahead_price",
                    "imbalance_price",
                    "mfrr_price",
                    "production",
                ]
                if c in df_match.columns
            ]

            if not value_cols:
                results[var_name] = df_match.to_dict(orient="records")
            else:
                results[var_name] = df_match[value_cols].to_dict(orient="records")

        else:
            print(
                f"⚠️ {filename} has neither 'prediction_for' nor 'time' as columns "
                f"(after index reset). Columns: {list(df.columns)}"
            )
            results[var_name] = None

    return results


def load_mmo_data(path):
    df = pd.read_parquet(path)

    print(df.head(6))

def load_parameters_from_parquet(time_str: str, scenarios: int, seed=None):
    print(f"\nLoading market data for time: {time_str}")
    data = load_market_data(
        time_str=time_str,
        raw_path="data",
        area="NO3",
        park="roan",
    )

    # Extract correct lists
    CM_up      = data["mfrr_cm_up_forecasts"]
    CM_down    = data["mfrr_cm_down_forecasts"]
    DA         = data["dayahead_forecasts"]
    EAM_up     = data["mfrr_eam_up_forecasts"]
    EAM_down   = data["mfrr_eam_down_forecasts"]
    wind_speed = data["production_forecasts"]
    
    # Goes through EAM_down and changes the sign of each value
    for i in range(len(EAM_down)):
        EAM_down[i] = -EAM_down[i]

    # We have added seed to be able to generate the same random numbers
    CM_up_sel, CM_down_sel, DA_sel, EAM_up_sel, EAM_down_sel, wind_speed_sel, picked_scenario_indices = select_scenarios(scenarios, CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed, seed)

    return CM_up_sel, CM_down_sel, DA_sel, EAM_up_sel, EAM_down_sel, wind_speed_sel, picked_scenario_indices


def get_global_bounds_from_raw_data(time_str: str):
    """
    Henter globale grenseverdier direkte fra rå markedsdata for et gitt tidspunkt.
    
    Dette uten å bygge hele scenario-treet eller velge ut scenarios.
    Brukes når du trenger Big-M konstanter før modelbygging.
    
    Args:
        time_str: Tidspunkt som string (f.eks. "2025-10-05")
    
    Returns:
        Dict med:
        - "Qmax": Høyeste produksjonkapasitet (wind speed) på tvers av alle forecasts
        - "Pmax": Høyeste pris på tvers av alle markeder og alle forecasts
        - "Pmax_per_market": Dict med høyeste pris per marked
                            Nøkler: "CM_up", "CM_down", "DA", "EAM_up", "EAM_down", "imbalance"
    """
    
    print(f"\nLoading global bounds for time: {time_str}")
    data = load_market_data(
        time_str=time_str,
        raw_path="data",
        area="NO3",
        park="roan",
    )
    
    # Samle alle priser
    all_prices = []
    Pmax_per_market = {}
    
    # CM markets
    if data["mfrr_cm_up_forecasts"]:
        cm_up_prices = [float(p) for p in data["mfrr_cm_up_forecasts"]]
        Pmax_per_market["CM_up"] = max(cm_up_prices)
        all_prices.extend(cm_up_prices)
    
    if data["mfrr_cm_down_forecasts"]:
        cm_down_prices = [float(p) for p in data["mfrr_cm_down_forecasts"]]
        Pmax_per_market["CM_down"] = max(cm_down_prices)
        all_prices.extend(cm_down_prices)
    
    # DA market
    if data["dayahead_forecasts"]:
        da_prices = [float(p) for p in data["dayahead_forecasts"]]
        Pmax_per_market["DA"] = max(da_prices)
        all_prices.extend(da_prices)
    
    # EAM markets
    if data["mfrr_eam_up_forecasts"]:
        eam_up_prices = [float(p) for p in data["mfrr_eam_up_forecasts"]]
        Pmax_per_market["EAM_up"] = max(eam_up_prices)
        all_prices.extend(eam_up_prices)
    
    if data["mfrr_eam_down_forecasts"]:
        eam_down_prices = [float(p) for p in data["mfrr_eam_down_forecasts"]]
        # EAM_down er negativ i dataene, så tar abs for å få maksimal verdi
        Pmax_per_market["EAM_down"] = max(abs(p) for p in eam_down_prices)
        all_prices.extend([abs(p) for p in eam_down_prices])
    
    # Imbalance prices (hvis tilgjengelig)
    if data["imbalance_forecasts"]:
        imb_prices = [float(p) for p in data["imbalance_forecasts"]]
        Pmax_per_market["imbalance"] = max(imb_prices)
        all_prices.extend(imb_prices)
    
    # Samle alle produksjonsverdier
    Qmax = 0
    if data["production_forecasts"]:
        wind_speeds = [float(w) for w in data["production_forecasts"]]
        Qmax = max(wind_speeds)
    
    # Finn høyeste pris på tvers av alle markeder
    Pmax = max(all_prices) if all_prices else 0
    
    global_bounds = {
        "Qmax": Qmax,
        "Pmax": Pmax,
        "Pmax_per_market": Pmax_per_market
    }
    
    print(f"✓ Global bounds computed:")
    print(f"  - Qmax: {Qmax:.4f}")
    print(f"  - Pmax: {Pmax:.4f}")
    print(f"  - Markets with data: {list(Pmax_per_market.keys())}")
    
    return global_bounds
