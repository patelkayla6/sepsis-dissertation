# eval_wrapper.py
from __future__ import annotations
import os, glob, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import importlib

CPCModel = importlib.import_module('model').CPCModel  # ← import here

# Re-use your existing code and helpers
from evaluation import (
    device,                                # uses your torch.device
    PPG_FEATURE_BASES,                     # feature schema
    CPCFeatureExtractor,                   # wraps CPC backbone
    load_train_stats,                      # loads imputer_med, scaler_mu, scaler_sd
    extract_features_for_paths,            # CPC-based windowed features
    extract_full_stats_for_paths_streaming,# streaming stats-only features
    build_feature_columns,                 # column names for DataFrame
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, recall_score
)

def _normalize_ids_index(idx: pd.Index) -> pd.Index:
    """Match the ID normalization you do in evaluation.py."""
    idx = idx.str.replace(r'^.*?-', '', regex=True)   # drop prefix like '01NVa-'
    idx = idx.str.replace(r'_.*$', '', regex=True)    # drop suffix like '_20s'
    return idx

def run_downstream_eval(
    ckpt_path: str = "cpc_best.pt",
    *,
    quick: bool = True,
    feat_mode: str = "cpc_mean",   # {'cpc_last','cpc_mean','stats','concat'}
    csv_glob_pattern: str = "../data/PPG/*.csv",
    labels_excel_path: str = "../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx",
    labels_sheet: str = "CLI",
    label_name: str = "DIED",
    remove_ids_for_died: list[str] | None = None,   # if None, uses your defaults when label_name == 'DIED'
    window_len: int = int(60 * 60 / 20),            # 180 (1h, 20s stride) — match your script
    use_streaming_full_stats: bool = False,         # only used if feat_mode == 'stats'
    cpc_hidden_dim: int = 256,
    cpc_proj_dim: int = 128,
    cpc_heads: int = 8,
    cpc_depth: int = 4,
    cpc_pred_steps: int = 1,
    stats_npz_path: str = "train_stats_ppg.npz",
    rf_estimators: int = 300,
    rf_max_depth: int | None = None,
    rf_random_state: int = 42,
) -> dict:
    """
    Programmatic downstream evaluation. Returns a dict with per-repeat arrays and summary means/stds.
    """

    # -------------------------
    # 1) Load CPC backbone
    # -------------------------
    D_INPUT = len(PPG_FEATURE_BASES)
    cpc = CPCModel(
        input_dim=D_INPUT,
        hidden_dim=cpc_hidden_dim,
        proj_dim=cpc_proj_dim,
        heads=cpc_heads,
        depth=cpc_depth,
        max_seq_len=window_len,
        pred_steps=cpc_pred_steps
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    cpc.load_state_dict(ckpt, strict=True)
    feature_model = CPCFeatureExtractor(cpc).to(device).eval()

    # -------------------------
    # 2) Load preprocess stats
    # -------------------------
    imputer_med, scaler_mu, scaler_sd = load_train_stats(stats_npz_path)

    # -------------------------
    # 3) Collect feature vectors
    # -------------------------
    csv_paths = sorted(glob.glob(csv_glob_pattern))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files matched: {csv_glob_pattern}")

    if feat_mode in ("cpc_last", "cpc_mean", "concat"):
        feats_np, ids = extract_features_for_paths(
            feature_model, csv_paths, PPG_FEATURE_BASES, window_len,
            imputer_med, scaler_mu, scaler_sd,
            mode=feat_mode
        )
    elif feat_mode == "stats":
        feats_np, ids = extract_full_stats_for_paths_streaming(
            csv_paths, PPG_FEATURE_BASES, chunksize=200_000 if use_streaming_full_stats else 50_000
        )
    else:
        raise ValueError(f"Unknown feat_mode: {feat_mode}")

    # -------------------------
    # 4) Build features DataFrame (columns depend on mode)
    # -------------------------
    columns = build_feature_columns(
        feat_mode, PPG_FEATURE_BASES, feats_np.shape[1],
        proj_dim_hint=cpc_proj_dim
    )
    features_df = pd.DataFrame(feats_np, index=ids, columns=columns)
    features_df.index = _normalize_ids_index(features_df.index)
    features_df.index.name = "USUBJID"

    # -------------------------
    # 5) Load labels and join
    # -------------------------
    excel = pd.ExcelFile(labels_excel_path)
    labels = pd.read_excel(excel, labels_sheet)[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
    labels = labels.astype({col: int for col in ['DIED', 'HFNOV', 'MEVEN']})
    labels = labels.set_index('USUBJID')

    dataset = features_df.join(labels, how='inner')

    # Optional removal to match your script’s behavior
    if label_name == 'DIED':
        default_ids = ['003-1011', '003-1016', '003-1017', '003-1214', '003-1219', '003-1239']
        to_remove = remove_ids_for_died if remove_ids_for_died is not None else default_ids
        dataset = dataset[~dataset.index.isin(to_remove)]

    X = dataset[columns].values
    y = dataset[label_name].values.astype(int)

    # -------------------------
    # 6) CV & classifier
    # -------------------------
    if quick:
        n_repeats, n_splits = 3, 5
    else:
        n_repeats, n_splits = 20, 10

    pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=True, with_std=True)),
        ("rf",  RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=rf_random_state
        ))
    ])

    aucs, auprcs, accs, sens_list, spec_list, f1s = [], [], [], [], [], []
    import numpy as _np
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rf_random_state + rep)
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            pipe.fit(Xtr, ytr)
            probs = pipe.predict_proba(Xte)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # metrics
            auc  = roc_auc_score(yte, probs)
            auprc = average_precision_score(yte, probs)
            acc  = (preds == yte).mean()
            sens = recall_score(yte, preds, pos_label=1)
            tn, fp, fn, tp = confusion_matrix(yte, preds, labels=[0,1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1   = f1_score(yte, preds)

            aucs.append(auc); auprcs.append(auprc)
            accs.append(acc); sens_list.append(sens); spec_list.append(spec); f1s.append(f1)

    summary = {
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "AUROC_mean": round(float(np.mean(aucs)), 3),
        "AUROC_std": round(float(np.std(aucs)), 3),
        "AUPRC_mean": round(float(np.mean(auprcs)), 3),
        "AUPRC_std": round(float(np.std(auprcs)), 3),
        "ACC_mean": round(float(np.mean(accs)), 3),
        "ACC_std": round(float(np.std(accs)), 3),
        "SENS_mean": round(float(np.mean(sens_list)), 3),
        "SENS_std": round(float(np.std(sens_list)), 3),
        "SPEC_mean": round(float(np.mean(spec_list)), 3),
        "SPEC_std": round(float(np.std(spec_list)), 3),
        "F1_mean": round(float(np.mean(f1s)), 3),
        "F1_std": round(float(np.std(f1s)), 3),
        # raw arrays (unrounded)
        "AUROC_all": aucs,
        "AUPRC_all": auprcs,
        "ACC_all": accs,
        "SENS_all": sens_list,
        "SPEC_all": spec_list,
        "F1_all": f1s,
    }
    return summary
