"""
feature_engineering.py
──────────────────────────────────────────────────────────────
Creates driver behaviour labels and model features.

Uses:
- trajectories
- weather
- air quality
- traffic lights

Task:
Driver behaviour classification
0 = safe
1 = moderate
2 = aggressive
"""

import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────
# THRESHOLDS
# ──────────────────────────────────────────────────────────────

THRESH = {
    "aggressive_accel":    2.0,
    "aggressive_velocity": 14.0,
    "aggressive_yaw":      15.0,

    "safe_accel":          0.5,
    "safe_velocity":       8.0,
    "safe_yaw":            5.0,
}

LABEL_MAP = {
    "safe": 0,
    "moderate": 1,
    "aggressive": 2,
}

LABEL_NAMES = {
    0: "safe",
    1: "moderate",
    2: "aggressive",
}

N_CLASSES = 3


# ╔════════════════════════════════════════════════════════════╗
# ║ FEATURE ENGINEERING                                       ║
# ╚════════════════════════════════════════════════════════════╝

def add_derived_features(df):

    df = df.copy()

    df = df.sort_values(["id", "timestamp"])

    # Yaw change
    df["yaw_change"] = (
        df.groupby("id")["yaw"]
        .diff()
        .abs()
        .fillna(0)
    )

    # Time features
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    df["sin_hour"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["cos_hour"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    return df


# ╔════════════════════════════════════════════════════════════╗
# ║ LABELLING                                                 ║
# ╚════════════════════════════════════════════════════════════╝

def label_behaviour(df):

    df = add_derived_features(df)

    is_aggressive = (
        (df["acceleration_magnitude"] > THRESH["aggressive_accel"]) |
        (df["velocity_magnitude"] > THRESH["aggressive_velocity"]) |
        (df["yaw_change"] > THRESH["aggressive_yaw"])
    )

    is_safe = (
        (df["acceleration_magnitude"] < THRESH["safe_accel"]) &
        (df["velocity_magnitude"] < THRESH["safe_velocity"]) &
        (df["yaw_change"] < THRESH["safe_yaw"])
    )

    df["behaviour_label"] = "moderate"

    df.loc[is_safe, "behaviour_label"] = "safe"

    df.loc[is_aggressive, "behaviour_label"] = "aggressive"

    df["behaviour_code"] = (
        df["behaviour_label"]
        .map(LABEL_MAP)
    )

    print("\n===== FULL DATASET DISTRIBUTION =====")

    counts = df["behaviour_label"].value_counts()

    for label, count in counts.items():

        print(
            f"{label:<12}: "
            f"{count:>10,} rows "
            f"({100*count/len(df):.1f}%)"
        )

    return df


# ╔════════════════════════════════════════════════════════════╗
# ║ VEHICLE SAMPLING                                          ║
# ╚════════════════════════════════════════════════════════════╝

def get_dominant_label(series):

    counts = {}

    for val in series:
        counts[val] = counts.get(val, 0) + 1

    return max(counts, key=counts.get)


def vehicle_stratified_sample(df,
                               vehicles_per_class=500,
                               random_state=42):
    """
    Vehicle-level stratified sampling.

    Keeps complete trajectories intact.
    """

    random.seed(random_state)

    print("\nComputing dominant behaviour per vehicle...")

    vehicle_labels = {}

    for vid, group in df.groupby("id"):

        vehicle_labels[vid] = get_dominant_label(
            group["behaviour_label"].tolist()
        )

    label_to_vehicles = {
        "safe": [],
        "moderate": [],
        "aggressive": [],
    }

    for vid, label in vehicle_labels.items():

        if label in label_to_vehicles:
            label_to_vehicles[label].append(vid)

    print("\nVehicles by dominant class:")

    for cls, vids in label_to_vehicles.items():

        print(f"{cls:<12}: {len(vids):>6,} vehicles")

    selected_vehicles = []

    for cls in ["safe", "moderate", "aggressive"]:

        vids = label_to_vehicles[cls]

        n = min(len(vids), vehicles_per_class)

        selected = random.sample(vids, n)

        selected_vehicles.extend(selected)

        print(f"Selected {n} {cls} vehicles")

    sampled = df[
        df["id"].isin(selected_vehicles)
    ].copy()

    sampled = (
        sampled
        .sort_values(["id", "timestamp"])
        .reset_index(drop=True)
    )

    print("\n===== VEHICLE-LEVEL SAMPLE =====")

    print(
        f"Vehicles selected : "
        f"{sampled['id'].nunique():,}"
    )

    print(
        f"Rows retained     : "
        f"{len(sampled):,}"
    )

    return sampled


# ╔════════════════════════════════════════════════════════════╗
# ║ SEQUENCE BUILDING                                         ║
# ╚════════════════════════════════════════════════════════════╝

def build_sequences(df,
                    feature_cols,
                    seq_len=30,
                    pred_horizon=10):
    """
    Build per-vehicle sequences.

    Predicts:
    behaviour class pred_horizon timesteps ahead.
    """

    X_list = []
    y_list = []

    vehicle_ids = df["id"].unique()

    print(
        f"\nBuilding sequences from "
        f"{len(vehicle_ids):,} vehicles..."
    )

    for vid in vehicle_ids:

        group = (
            df[df["id"] == vid]
            .sort_values("timestamp")
        )

        if len(group) <= seq_len + pred_horizon:
            continue

        features = group[feature_cols].values

        labels = group["behaviour_code"].values

        for i in range(
            len(group) - seq_len - pred_horizon
        ):

            X_list.append(
                features[i:i + seq_len]
            )

            target_idx = (
                i + seq_len + pred_horizon
            )

            y_list.append(
                labels[target_idx]
            )

    if not X_list:

        raise ValueError(
            "No sequences built."
        )

    print(f"Built {len(X_list):,} sequences")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64)
    )


# ╔════════════════════════════════════════════════════════════╗
# ║ FEATURE LISTS                                             ║
# ╚════════════════════════════════════════════════════════════╝

def get_feature_columns(df):

    base = [
        "velocity_magnitude",
        "acceleration_magnitude",
        "yaw_change",
        "sin_hour",
        "cos_hour",
    ]

    weather_feats = [
        "air_temperature",
        "wind_speed",
        "rain_intensity",
        "relative_humidity",
        "visibility",
    ]

    aq_feats = [
        "no2_gas_concentration",
        "co_gas_concentration",
    ]

    signal_feats = [
        "signal_phase_code",
    ]

    return (
        base +
        [f for f in weather_feats if f in df.columns] +
        [f for f in aq_feats if f in df.columns] +
        [f for f in signal_feats if f in df.columns]
    )


def get_feature_columns_leakage_reduced(df):
    """
    Removes variables directly used
    to create labels.
    """

    feature_cols = [

        # Time
        "sin_hour",
        "cos_hour",

        # Weather
        "air_temperature",
        "wind_speed",
        "rain_intensity",
        "relative_humidity",
        "visibility",

        # Air quality
        "no2_gas_concentration",
        "co_gas_concentration",

        # Traffic lights
        "signal_phase_code",
    ]

    return [
        c for c in feature_cols
        if c in df.columns
    ]


# ╔════════════════════════════════════════════════════════════╗
# ║ PREPARE MODEL DATA                                        ║
# ╚════════════════════════════════════════════════════════════╝

def prepare_model_data(df,
                       seq_len=30,
                       test_size=0.2,
                       vehicles_per_class=500):

    df = label_behaviour(df)

    df = vehicle_stratified_sample(
        df,
        vehicles_per_class=vehicles_per_class
    )

    feature_cols = get_feature_columns(df)

    print(f"\n===== FEATURES ({len(feature_cols)}) =====")

    for f in feature_cols:
        print(f"• {f}")

    # ──────────────────────────────────────────────────────
    # 80/20 split BEFORE sequence generation
    # ──────────────────────────────────────────────────────

    vehicle_ids = df["id"].unique()

    np.random.seed(42)

    np.random.shuffle(vehicle_ids)

    split_idx = int(
        len(vehicle_ids) * (1 - test_size)
    )

    train_ids = vehicle_ids[:split_idx]

    test_ids = vehicle_ids[split_idx:]

    train_df = (
        df[df["id"].isin(train_ids)]
        .copy()
    )

    test_df = (
        df[df["id"].isin(test_ids)]
        .copy()
    )

    # ──────────────────────────────────────────────────────
    # Scale ONLY using train rows
    # ──────────────────────────────────────────────────────

    scaler = StandardScaler()

    scaler.fit(
        train_df[feature_cols]
    )

    train_df[feature_cols] = scaler.transform(
        train_df[feature_cols]
    )

    test_df[feature_cols] = scaler.transform(
        test_df[feature_cols]
    )

    # ──────────────────────────────────────────────────────
    # Build sequences separately
    # ──────────────────────────────────────────────────────

    X_train, y_train = build_sequences(
        train_df,
        feature_cols,
        seq_len=seq_len,
        pred_horizon=10
    )

    X_test, y_test = build_sequences(
        test_df,
        feature_cols,
        seq_len=seq_len,
        pred_horizon=10
    )

    print("\n===== FINAL DATA =====")

    print(f"Train sequences : {X_train.shape}")

    print(f"Test sequences  : {X_test.shape}")

    print(f"80/20 split     : OK")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_cols,
        df
    )

if __name__ == "__main__":

    print("\nfeature_engineering.py loaded successfully.")
    print("All functions compiled correctly.")