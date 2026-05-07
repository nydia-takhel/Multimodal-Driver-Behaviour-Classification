"""
loader.py
─────────────────────────────────────────────────────────────
Loads all 4 DLR Urban Traffic Dataset folders.

Dataset: DLR-UT v1.0.0
Location: AIM Research Intersection, Braunschweig, Germany
Coverage: 24.09.2023 00:00:00 – 25.09.2023 00:00:00 UTC

Sampling rates (from documentation):
  Trajectory data : 20 Hz
  Traffic lights  : 1 Hz
  Weather         : 10 / 30 / 60 seconds depending on column
  Air quality     : 60 seconds (gas concentrations)

Merge strategy:
  All 4 datasets cover the same day in 15-minute batches.
  When loading the full dataset, all timestamps overlap.
  merge_asof is used to join datasets by nearest timestamp.

Traffic light state codes (Table 3, DLR documentation):
  1 = dark              (light is off)
  3 = stop-And-Remain   (RED)
  4 = pre-Movement      (transitioning red → green)
  5 = permissive-Movement-Allowed  (GREEN)
  7 = permissive-clearance         (YELLOW)
  9 = caution-Conflicting-Traffic  (YELLOW flashing)
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Dataset path — update this to your local path ─────────────────────────────
# BASE_PATH = "/home/bel/Desktop/nydia/PHASE 2/DLR_Urban_Traffic_dataset_v1-0-0"
# BASE_PATH = "/home/bel/Desktop/nydia/driver behaviour/DLR-UT_v1-0-0"
BASE_PATH = "/home/bel/Desktop/nydia/driver behaviour/DLR-UT_v1-0-0"

# ── Traffic light state map (from official DLR documentation, Table 3) ────────
TL_STATE_MAP = {
    0: "unavailable",
    1: "dark",
    2: "stop_then_proceed",
    3: "red",
    4: "pre_movement",
    5: "green",
    6: "protected_green",
    7: "yellow",
    8: "protected_yellow",
    9: "yellow_flash",
}

# Grouped for analysis
TL_PHASE_MAP = {
    "red":              "red",
    "stop_then_proceed":"red",
    "pre_movement":     "amber",
    "green":            "green",
    "protected_green":  "green",
    "yellow":           "amber",
    "protected_yellow": "amber",
    "yellow_flash":     "amber",
    "dark":             "off",
    "unavailable":      "off",
}


def _load_folder(folder_name, extensions=("*.csv",)):
    """Load all files from a dataset folder into a single DataFrame."""
    folder_path = os.path.join(BASE_PATH, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"\n❌ Folder not found: {folder_path}"
            f"\n   Please check BASE_PATH in loader.py"
            f"\n   Current BASE_PATH: {BASE_PATH}"
        )

    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(folder_path, ext))
    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    print(f"  📂 {folder_name}: loading {len(files)} files...")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"     ⚠️  Skipped {os.path.basename(f)}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"     ✅ {len(df):,} rows loaded")
    return df


def _parse_timestamps(df):
    """Parse timestamp column to timezone-aware UTC datetime."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _fill_missing(df):
    """
    Fill ALL missing values — no data is removed.
    Strategy: forward fill → backward fill → column mean.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill()
        .bfill()
        .fillna(df[numeric_cols].mean())
    )
    return df


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_trajectories():
    """
    Load all trajectory files.
    Adds derived column: vehicle_type (dominant classification per row).
    Sampling: 20 Hz (~50ms between rows per vehicle)
    """
    df = _load_folder("trajectories")
    df = _parse_timestamps(df)
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    df = _fill_missing(df)

    # Derive dominant vehicle type per row
    type_cols = [
        "classifications_pedestrian", "classifications_bicycle",
        "classifications_motorbike", "classifications_car",
        "classifications_van", "classifications_truck"
    ]
    type_labels = ["pedestrian", "bicycle", "motorbike", "car", "van", "truck"]
    df["vehicle_type"] = df[type_cols].idxmax(axis=1).map(
        dict(zip(type_cols, type_labels))
    )

    print(f"     🚗 {df['id'].nunique():,} unique vehicles | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_weather():
    """
    Load all weather files.
    Sampling: 10s (most columns), 30s (surface/rain), 60s (solar).
    """
    df = _load_folder("weather")
    df = _parse_timestamps(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _fill_missing(df)

    # Keep useful columns only
    keep = [
        "timestamp", "air_temperature", "relative_humidity",
        "wind_speed", "wind_direction", "visibility",
        "rain_intensity", "air_pressure_msl",
        "surface_state", "surface_grip"
    ]
    df = df[[c for c in keep if c in df.columns]]

    print(f"     🌤️  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_air_quality():
    """
    Load all air quality files.
    Sampling: 60 seconds for gas concentrations.
    All sensors are at the same intersection location.
    """
    df = _load_folder("air_quality")
    df = _parse_timestamps(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _fill_missing(df)

    print(f"     🌫️  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_traffic_lights():
    """
    Load all traffic light files.
    Sampling: 1 Hz.
    30 unique traffic lights at the intersection.
    States mapped using official DLR documentation Table 3.
    """
    df = _load_folder("traffic_lights")
    df = _parse_timestamps(df)
    df = df.sort_values(["timestamp", "id"]).reset_index(drop=True)

    # Apply correct state labels from documentation
    df["state_name"]  = df["state"].map(TL_STATE_MAP).fillna("unknown")
    df["state_phase"] = df["state_name"].map(TL_PHASE_MAP).fillna("unknown")

    print(f"     🚦 {df['id'].nunique()} signals | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ── Aggregation helpers ────────────────────────────────────────────────────────

def aggregate_traffic_lights(tl_df, freq="1s"):
    """
    Aggregate traffic light states to a single row per timestamp.
    For each timestamp, compute:
      - dominant phase across all 30 signals
      - % of signals in each phase
    """
    tl_df = tl_df.copy()
    tl_df = tl_df.set_index("timestamp")
    resampled = tl_df.groupby(pd.Grouper(freq=freq))["state_phase"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "unknown"
    ).reset_index()
    resampled.columns = ["timestamp", "dominant_signal_phase"]

    # Encode as numeric for model features
    phase_enc = {"red": 0, "amber": 1, "green": 2, "off": 3, "unknown": -1}
    resampled["signal_phase_code"] = resampled["dominant_signal_phase"].map(phase_enc)
    return resampled


def aggregate_air_quality(aq_df, freq="1min"):
    """
    Resample air quality to regular intervals.
    """
    aq_df = aq_df.copy().set_index("timestamp")
    numeric = aq_df.select_dtypes(include=[np.number])
    resampled = numeric.resample(freq).mean().ffill().bfill().reset_index()
    return resampled


# ── Master merge function ──────────────────────────────────────────────────────

def merge_all_datasets(traj, weather, aq, tl):
    """
    Merge all 4 datasets into a single DataFrame using merge_asof.

    Strategy:
      1. Aggregate traffic lights to 1 dominant state per second
      2. Resample air quality to 1-minute intervals
      3. merge_asof trajectory ← weather (10s tolerance)
      4. merge_asof result ← air quality (60s tolerance)
      5. merge_asof result ← traffic lights (2s tolerance)

    NO rows are removed. Unmatched rows get NaN → filled.
    """
    print("\n  Aggregating traffic lights...")
    tl_agg = aggregate_traffic_lights(tl, freq="1s")

    print("  Resampling air quality...")
    aq_agg = aggregate_air_quality(aq, freq="1min")

    print("  Merging trajectory + weather...")
    traj_s   = traj.sort_values("timestamp").reset_index(drop=True)
    weather_s = weather.sort_values("timestamp").reset_index(drop=True)

    df = pd.merge_asof(
        traj_s, weather_s,
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("10s")
    )

    print("  Merging + air quality...")
    aq_agg_s = aq_agg.sort_values("timestamp").reset_index(drop=True)
    df = pd.merge_asof(
        df, aq_agg_s,
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("60s")
    )

    print("  Merging + traffic lights...")
    tl_agg_s = tl_agg.sort_values("timestamp").reset_index(drop=True)
    df = pd.merge_asof(
        df, tl_agg_s,
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("2s")
    )

    # Fill any remaining NaNs
    df = _fill_missing(df)

    nan_count = df.isna().sum().sum()
    print(f"\n  ✅ Final dataset: {len(df):,} rows | "
          f"{df.shape[1]} columns | {nan_count} NaNs")
    return df


def get_dataset_summary(df, aq_df, tl_df):
    """Print a summary of the merged dataset for reporting."""
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Intersection : Hans-Sommer-Str., Braunschweig, Germany")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Vehicles     : {df['id'].nunique():,}")
    print(f"  Time range   : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"\n  Air Quality (mean ambient):")
    for col in ["no2_gas_concentration", "co_gas_concentration",
                "fine_particle_mass_concentration"]:
        if col in df.columns:
            print(f"    {col}: {df[col].mean():.2f}")
    print(f"\n  Traffic Light Phase Distribution:")
    if "dominant_signal_phase" in df.columns:
        for phase, count in df["dominant_signal_phase"].value_counts().items():
            print(f"    {phase:<12}: {100*count/len(df):.1f}%")
    print("="*55)


if __name__ == "__main__":
    print("\n===== LOADING DLR URBAN TRAFFIC DATASET =====\n")

    traj    = load_trajectories()
    weather = load_weather()
    aq      = load_air_quality()
    tl      = load_traffic_lights()

    df = merge_all_datasets(traj, weather, aq, tl)
    get_dataset_summary(df, aq, tl)

    print("\nSample:")
    print(df[["timestamp", "id", "vehicle_type", "velocity_magnitude",
              "air_temperature", "no2_gas_concentration",
              "dominant_signal_phase"]].head())
