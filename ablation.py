"""
ablation.py
──────────────────────────────────────────────────────────────────
Feature Ablation Study

Tests whether each data modality genuinely contributes to
classification performance beyond trajectory features alone.

Four conditions, same LSTM, same split, same hyperparameters:

  1. Trajectory only        (3 features)
     velocity, acceleration, yaw_change

  2. Trajectory + Time      (5 features)
     adds sin_hour, cos_hour

  3. Trajectory + Time + Weather  (10 features)
     adds temperature, wind, rain, humidity, visibility

  4. All features           (13 features)
     adds no2, co, signal_phase_code

Same vehicle-level 80/20 split, same pred_horizon=10.
Any accuracy difference is attributable solely to the added features.
"""

import sys, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import (load_trajectories, load_weather,
                    load_air_quality, load_traffic_lights,
                    merge_all_datasets)
from feature_engineering import (label_behaviour,
                                  vehicle_stratified_sample,
                                  build_sequences)
from trainer import LSTMClassifier, DEVICE

SEQ_LEN      = 30
PRED_HORIZON = 10
LABEL_NAMES  = ['safe', 'moderate', 'aggressive']

FEATURE_SETS = {
    'Traj only (3)': [
        'velocity_magnitude',
        'acceleration_magnitude',
        'yaw_change',
    ],
    'Traj + Time (5)': [
        'velocity_magnitude', 'acceleration_magnitude', 'yaw_change',
        'sin_hour', 'cos_hour',
    ],
    'Traj + Time + Weather (10)': [
        'velocity_magnitude', 'acceleration_magnitude', 'yaw_change',
        'sin_hour', 'cos_hour',
        'air_temperature', 'wind_speed', 'rain_intensity',
        'relative_humidity', 'visibility',
    ],
    'All Features (13)': [
        'velocity_magnitude', 'acceleration_magnitude', 'yaw_change',
        'sin_hour', 'cos_hour',
        'air_temperature', 'wind_speed', 'rain_intensity',
        'relative_humidity', 'visibility',
        'no2_gas_concentration', 'co_gas_concentration',
        'signal_phase_code',
    ],
}


def make_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_lstm(n_feat, X_train, y_train, X_val, y_val,
               epochs=30, batch_size=256, lr=1e-3):
    model     = LSTMClassifier(n_feat).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=5, factor=0.5
    )
    best_val  = float('inf')
    best_w    = None
    no_impr   = 0

    train_loader = make_loader(X_train, y_train, batch_size, True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, False)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_w   = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
            no_impr  = 0
        else:
            no_impr += 1
            if no_impr >= 5:
                break

    if best_w:
        model.load_state_dict(best_w)
    return model


def evaluate(model, X_test, y_test, batch_size=512):
    model.eval()
    preds = []
    dl = make_loader(X_test, y_test, batch_size, False)
    with torch.no_grad():
        for xb, _ in dl:
            preds.extend(
                model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            )
    preds   = np.array(preds)
    acc     = accuracy_score(y_test, preds)
    f1_w    = f1_score(y_test, preds, average='weighted')
    f1_per  = f1_score(y_test, preds, average=None,
                       labels=[0,1,2], zero_division=0)
    return acc, f1_w, f1_per


def main():
    print("\n" + "="*62)
    print("  SADAK — FEATURE ABLATION STUDY")
    print("  Model: LSTM (fixed architecture)")
    print("  Pred horizon: 10 steps (500ms ahead)")
    print("  Split: vehicle-level 80/20")
    print("="*62)

    # ── Load once ──────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    traj    = load_trajectories()
    weather = load_weather()
    aq      = load_air_quality()
    tl      = load_traffic_lights()
    df      = merge_all_datasets(traj, weather, aq, tl)
    df      = label_behaviour(df)
    df      = vehicle_stratified_sample(df, vehicles_per_class=500)

    # ── Vehicle-level split (same across all conditions) ───────
    vehicle_ids = df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(vehicle_ids)
    split_idx     = int(len(vehicle_ids) * 0.80)
    train_ids_all = vehicle_ids[:split_idx]
    test_ids      = vehicle_ids[split_idx:]
    val_split     = int(len(train_ids_all) * 0.90)
    train_ids     = train_ids_all[:val_split]
    val_ids       = train_ids_all[val_split:]

    train_df = df[df['id'].isin(train_ids)].copy()
    val_df   = df[df['id'].isin(val_ids)].copy()
    test_df  = df[df['id'].isin(test_ids)].copy()

    print(f"  Train: {len(train_ids)} vehicles  "
          f"Val: {len(val_ids)} vehicles  "
          f"Test: {len(test_ids)} vehicles")

    results = []

    print("\n[2/3] Running ablation conditions...")

    for cond_name, feat_cols in FEATURE_SETS.items():
        # Only use features present in df
        feat_cols = [f for f in feat_cols if f in df.columns]
        n_feat    = len(feat_cols)

        print(f"\n── {cond_name} ──")

        # Scale on train only
        scaler = StandardScaler()
        scaler.fit(train_df[feat_cols])

        tr = train_df.copy(); tr[feat_cols] = scaler.transform(tr[feat_cols])
        vl = val_df.copy();   vl[feat_cols] = scaler.transform(vl[feat_cols])
        te = test_df.copy();  te[feat_cols] = scaler.transform(te[feat_cols])

        X_train, y_train = build_sequences(tr, feat_cols,
                                            seq_len=SEQ_LEN,
                                            pred_horizon=PRED_HORIZON)
        X_val,   y_val   = build_sequences(vl, feat_cols,
                                            seq_len=SEQ_LEN,
                                            pred_horizon=PRED_HORIZON)
        X_test,  y_test  = build_sequences(te, feat_cols,
                                            seq_len=SEQ_LEN,
                                            pred_horizon=PRED_HORIZON)

        t0    = time.time()
        model = train_lstm(n_feat, X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0

        acc, f1_w, f1_per = evaluate(model, X_test, y_test)

        print(f"   Accuracy  : {acc*100:.2f}%")
        print(f"   F1 (wtd)  : {f1_w*100:.2f}%")
        print(f"   F1 safe   : {f1_per[0]*100:.2f}%  "
              f"moderate: {f1_per[1]*100:.2f}%  "
              f"aggressive: {f1_per[2]*100:.2f}%")
        print(f"   Time      : {elapsed:.0f}s")

        results.append({
            'condition':   cond_name,
            'n_features':  n_feat,
            'accuracy':    round(acc*100, 2),
            'f1_weighted': round(f1_w*100, 2),
            'f1_safe':     round(f1_per[0]*100, 2),
            'f1_moderate': round(f1_per[1]*100, 2),
            'f1_agg':      round(f1_per[2]*100, 2),
        })

    # ── Summary ────────────────────────────────────────────────
    print("\n\n" + "="*72)
    print("  ABLATION RESULTS")
    print("="*72)
    print(f"  {'Condition':<32} {'Acc':>7} {'F1':>7} "
          f"{'F1-Safe':>8} {'F1-Mod':>8} {'F1-Agg':>8}")
    print(f"  {'-'*68}")

    base_acc = None
    for r in results:
        if base_acc is None:
            base_acc = r['accuracy']
            delta = ''
        else:
            diff  = r['accuracy'] - base_acc
            delta = f"  (+{diff:.2f}pp)" if diff >= 0 else f"  ({diff:.2f}pp)"
        print(f"  {r['condition']:<32} "
              f"{r['accuracy']:>6.2f}% "
              f"{r['f1_weighted']:>6.2f}% "
              f"{r['f1_safe']:>7.2f}% "
              f"{r['f1_moderate']:>7.2f}% "
              f"{r['f1_agg']:>7.2f}%"
              f"{delta}")

    # ── Verdict ─────────────────────────────────────────────────
    traj_acc = results[0]['accuracy']
    full_acc = results[-1]['accuracy']
    gain     = full_acc - traj_acc

    print(f"\n  Trajectory only : {traj_acc:.2f}%")
    print(f"  All features    : {full_acc:.2f}%")
    print(f"  Total gain      : {gain:+.2f}pp")

    if gain > 2.0:
        print(f"  Verdict: Multi-modal features ADD {gain:.2f}pp — claim JUSTIFIED")
    elif gain > 0.5:
        print(f"  Verdict: Multi-modal features add MODEST {gain:.2f}pp — marginal gain")
    else:
        print(f"  Verdict: Multi-modal features add NEGLIGIBLE {gain:.2f}pp — "
              f"trajectory alone is sufficient")

    import json
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'ablation_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: ablation_results.json")


if __name__ == '__main__':
    main()