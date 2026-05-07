"""
baseline_eval.py
─────────────────────────────────────────────────────────────
Fix 1: Proper evaluation with baselines and non-overlapping windows.

Runs four evaluations:
  A. Naive baseline (predict majority class always)
  B. Logistic Regression (no sequential context — flat features)
  C. Deep learning models on OVERLAPPING windows (stride=1)
  D. Deep learning models on NON-OVERLAPPING windows (stride=seq_len)

Comparing C vs D shows the true impact of sliding window overlap.
Comparing B vs C shows whether sequential architecture adds value.
Comparing A vs B shows whether features matter at all.

Without a baseline, 99% accuracy is a headline. With these
comparisons, 99% becomes a finding.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)
from sklearn.preprocessing import StandardScaler
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import (load_trajectories, load_weather,
                    load_air_quality, load_traffic_lights,
                    merge_all_datasets)
from feature_engineering import (label_behaviour,
                                  vehicle_stratified_sample,
                                  get_feature_columns)
from trainer import (RNNClassifier, LSTMClassifier,
                     TransformerClassifier, DEVICE)

LABEL_NAMES = ['safe', 'moderate', 'aggressive']
SEQ_LEN     = 30
SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))


# ── Sequence builders ──────────────────────────────────────────

def build_sequences_strided(df, feature_cols, seq_len=30, stride=1, pred_horizon=10):
    """
    Build sequences with configurable stride.

    stride=1        : overlapping — consecutive sequences share (seq_len-1) timesteps.
    stride=seq_len  : non-overlapping — no shared timesteps, honest evaluation.
    pred_horizon=10 : matches trainer — predicts label 500ms ahead (not immediate next step).
    """
    X_list, y_list = [], []
    for vid in df['id'].unique():
        group = df[df['id'] == vid].sort_values('timestamp')
        if len(group) <= seq_len + pred_horizon:
            continue
        data   = group[feature_cols].values.astype(np.float32)
        labels = group['behaviour_code'].values.astype(np.int64)
        for i in range(0, len(group) - seq_len - pred_horizon, stride):
            X_list.append(data[i:i+seq_len])
            y_list.append(labels[i+seq_len+pred_horizon])
    if not X_list:
        raise ValueError(f"No sequences built with stride={stride}")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ── Baseline A: Majority class ─────────────────────────────────

def majority_class_baseline(y_train, y_test):
    """
    Predicts the most frequent class in training set for every sample.
    This is the absolute floor — any real model must beat this.
    """
    majority = int(np.bincount(y_train).argmax())
    y_pred   = np.full_like(y_test, majority)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cls_name = LABEL_NAMES[majority]
    print(f"\n── Baseline A: Majority Class (always predict '{cls_name}') ──")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   F1 Score : {f1*100:.2f}%")
    print(f"   ⚠ This is the floor — any model must beat this to be meaningful")
    return {'model': 'Majority Class', 'accuracy': round(acc*100,2),
            'f1': round(f1*100,2), 'note': f'always predict {cls_name}'}


# ── Baseline B: Logistic Regression ───────────────────────────

def logistic_regression_baseline(X_train, X_test, y_train, y_test):
    """
    Logistic Regression on FLATTENED sequences.
    No temporal structure — treats the 30×13 window as 390 independent features.

    If this matches deep learning accuracy, the sequence models are not
    learning temporal patterns — they are learning feature distributions.
    The gap between LR and LSTM tells you how much the temporal
    architecture actually contributes.
    """
    print(f"\n── Baseline B: Logistic Regression (no temporal context) ──")
    print(f"   Flattening sequences: {X_train.shape} → {(X_train.shape[0], X_train.shape[1]*X_train.shape[2])}")

    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_te_flat = X_test.reshape(len(X_test),  -1)

    # Scale flat features
    sc = StandardScaler()
    X_tr_flat = sc.fit_transform(X_tr_flat)
    X_te_flat = sc.transform(X_te_flat)

    print("   Training Logistic Regression...")
    lr = LogisticRegression(max_iter=500, C=1.0, solver='saga',
                             n_jobs=-1, verbose=0)
    lr.fit(X_tr_flat, y_train)
    y_pred = lr.predict(X_te_flat)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   F1 Score : {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))
    return {'model': 'Logistic Regression', 'accuracy': round(acc*100,2),
            'f1': round(f1*100,2), 'note': 'flat features, no temporal context'}


# ── Evaluate saved deep learning model ────────────────────────

def eval_dl_model(name, model, X_test, y_test, batch_size=512):
    model = model.to(DEVICE)
    model.eval()
    preds = []
    ds = TensorDataset(torch.tensor(X_test),
                       torch.tensor(y_test, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for xb, _ in dl:
            preds.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')
    return acc, f1, preds


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  SADAK — BASELINE EVALUATION & NON-OVERLAPPING ASSESSMENT")
    print("="*65)

    # ── 1. Load data ────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    traj    = load_trajectories()
    weather = load_weather()
    aq      = load_air_quality()
    tl      = load_traffic_lights()
    df      = merge_all_datasets(traj, weather, aq, tl)

    # ── 2. Label + sample ───────────────────────────────────────
    print("\n[2/5] Labelling & sampling...")
    df = label_behaviour(df)
    df = vehicle_stratified_sample(df, vehicles_per_class=500)
    feature_cols = get_feature_columns(df)
    n_feat = len(feature_cols)

    # Scale on first 80% only
    n_train = int(len(df) * 0.8)
    scaler  = StandardScaler()
    scaler.fit(df.iloc[:n_train][feature_cols])
    df_sc = df.copy()
    df_sc[feature_cols] = scaler.transform(df[feature_cols])

    # ── 3. Build BOTH sequence sets ─────────────────────────────
    print("\n[3/5] Building sequences...")
    print("  Building overlapping sequences (stride=1)...")
    X_ov, y_ov   = build_sequences_strided(df_sc, feature_cols,
                                             SEQ_LEN, stride=1)
    sp_ov         = int(len(X_ov) * 0.8)
    X_tr_ov, X_te_ov = X_ov[:sp_ov], X_ov[sp_ov:]
    y_tr_ov, y_te_ov = y_ov[:sp_ov], y_ov[sp_ov:]
    print(f"  Overlapping    → {len(X_ov):,} total | "
          f"train: {len(X_tr_ov):,} | test: {len(X_te_ov):,}")

    print("  Building non-overlapping sequences (stride=30)...")
    X_no, y_no   = build_sequences_strided(df_sc, feature_cols,
                                             SEQ_LEN, stride=SEQ_LEN)
    sp_no         = int(len(X_no) * 0.8)
    X_tr_no, X_te_no = X_no[:sp_no], X_no[sp_no:]
    y_tr_no, y_te_no = y_no[:sp_no], y_no[sp_no:]
    print(f"  Non-overlapping → {len(X_no):,} total | "
          f"train: {len(X_tr_no):,} | test: {len(X_te_no):,}")
    print(f"\n  ⚠  Overlap inflates sequence count by "
          f"{len(X_ov)/max(len(X_no),1):.0f}× "
          f"({len(X_ov):,} vs {len(X_no):,})")

    # ── 4. Baselines ────────────────────────────────────────────
    print("\n[4/5] Running baselines...")
    results = []

    # A. Majority class
    r = majority_class_baseline(y_tr_ov, y_te_ov)
    results.append(r)

    # B. Logistic Regression
    r = logistic_regression_baseline(X_tr_ov, X_te_ov, y_tr_ov, y_te_ov)
    results.append(r)

    # ── 5. Evaluate saved DL models ─────────────────────────────
    print("\n[5/5] Evaluating saved deep learning models...")
    model_cfgs = [
        ('RNN',         RNNClassifier(n_feat)),
        ('LSTM',        LSTMClassifier(n_feat)),
        ('Transformer', TransformerClassifier(n_feat, seq_len=SEQ_LEN)),
    ]

    for name, model in model_cfgs:
        path = os.path.join(SAVE_DIR, f"{name.lower()}_model.pt")
        if not os.path.exists(path):
            print(f"  ⚠ {name}: model file not found — run trainer.py first")
            continue
        # Detect seq_len from checkpoint to handle mismatches
        ckpt = torch.load(path, map_location=DEVICE)
        if name == 'Transformer' and 'pos_embedding.weight' in ckpt:
            saved_seq_len = ckpt['pos_embedding.weight'].shape[0]
            if saved_seq_len != SEQ_LEN:
                print(f"  ℹ {name}: checkpoint seq_len={saved_seq_len}, rebuilding model")
                model = TransformerClassifier(n_feat, seq_len=saved_seq_len)
        model.load_state_dict(ckpt)

        # Overlapping eval
        acc_ov, f1_ov, _ = eval_dl_model(name, model, X_te_ov, y_te_ov)
        # Non-overlapping eval
        acc_no, f1_no, preds_no = eval_dl_model(name, model,
                                                  X_te_no, y_te_no)

        print(f"\n── {name} ──")
        print(f"   Overlapping windows   : {acc_ov*100:.2f}% acc | {f1_ov*100:.2f}% F1")
        print(f"   Non-overlapping windows: {acc_no*100:.2f}% acc | {f1_no*100:.2f}% F1")
        print(f"   Accuracy drop from overlap: {(acc_ov-acc_no)*100:.2f} pp")
        print(classification_report(y_te_no, preds_no,
                                    target_names=LABEL_NAMES, digits=4))

        results.append({'model': f'{name} (overlapping)',
                        'accuracy': round(acc_ov*100,2),
                        'f1': round(f1_ov*100,2),
                        'note': 'stride=1, shared timesteps'})
        results.append({'model': f'{name} (non-overlapping)',
                        'accuracy': round(acc_no*100,2),
                        'f1': round(f1_no*100,2),
                        'note': 'stride=30, independent sequences'})

    # ── Final table ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("  COMPLETE RESULTS TABLE")
    print("="*65)
    print(f"  {'Model':<35} {'Accuracy':>10} {'F1':>8}  Note")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['model']:<35} {r['accuracy']:>9.2f}% {r['f1']:>7.2f}%  {r['note']}")

    print(f"\n  KEY QUESTIONS ANSWERED:")
    baseline_acc = next((r['accuracy'] for r in results
                        if 'Majority' in r['model']), None)
    lr_acc       = next((r['accuracy'] for r in results
                        if 'Logistic' in r['model']), None)
    lstm_ov      = next((r['accuracy'] for r in results
                        if 'LSTM (over' in r['model']), None)
    lstm_no      = next((r['accuracy'] for r in results
                        if 'LSTM (non' in r['model']), None)

    if all([baseline_acc, lr_acc, lstm_ov, lstm_no]):
        print(f"  Do features matter?  "
              f"LR ({lr_acc:.1f}%) vs Majority ({baseline_acc:.1f}%) "
              f"= +{lr_acc-baseline_acc:.1f}pp ✓")
        print(f"  Does LSTM > LR?      "
              f"LSTM ({lstm_no:.1f}%) vs LR ({lr_acc:.1f}%) "
              f"= +{lstm_no-lr_acc:.1f}pp "
              f"{'✓ sequence matters' if lstm_no>lr_acc+2 else '⚠ marginal gain'}")
        print(f"  Window overlap impact: "
              f"{lstm_ov:.1f}% → {lstm_no:.1f}% "
              f"({lstm_ov-lstm_no:.1f}pp inflation from overlap)")

    # Save results
    import json
    with open(os.path.join(SAVE_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to baseline_results.json")


if __name__ == '__main__':
    main()