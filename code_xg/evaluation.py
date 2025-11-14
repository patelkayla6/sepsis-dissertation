import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, f1_score, recall_score
)
import importlib
CPCModel = importlib.import_module('model').CPCModel  # ← import here

# ---------------------------------------------------------------------
# Import your trained CPC backbone (RoPE-enabled) from another module
# ---------------------------------------------------------------------
# Change this to the correct module path if needed.

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# Feature list (ensure the comma after HRV_LZC)
# ---------------------------------------------------------------------
PPG_FEATURE_BASES = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
    'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
    'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN',
    'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_HF',
    'HRV_VHF', 'HRV_TP', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
    'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PAS',
    'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a',
    'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd',
    'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta',
    'HRV_MFDFA_alpha1_Asymmetry', 'HRV_MFDFA_alpha1_Fluctuation',
    'HRV_MFDFA_alpha1_Increment', 'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn',
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC',
    'Tpi', 'Tpp', 'Tsys', 'Tdia', 'Tsp', 'Tdp', 'deltaT',
     'Tsw10', 'Tsw25', 'Tsw33', 'Tsw50', 'Tsw66', 'Tsw75', 'Tsw90',
     'Tdw10', 'Tdw25', 'Tdw33', 'Tdw50', 'Tdw66', 'Tdw75', 'Tdw90',
     'Tpw10', 'Tpw25', 'Tpw33', 'Tpw50', 'Tpw66', 'Tpw75', 'Tpw90',
     'Asp', 'Adn', 'Adp', 'Aoff', 'AUCpi', 'AUCsys', 'AUCdia', ]

# ---------------------------------------------------------------------
# Wrapper: use CPCModel as a feature extractor
# ---------------------------------------------------------------------
class CPCFeatureExtractor(nn.Module):
    """
    Wraps a CPCModel and exposes per-timestep projected features.
    Works whether CPCModel had absolute PE or RoPE (auto-detects pos_encoding).
    """
    def __init__(self, cpc: CPCModel):
        super().__init__()
        self.cpc = cpc
        self.has_abs_pos = hasattr(cpc, 'pos_encoding')

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]  -> returns z: [B, T, P]
        """
        B, T, D = x.shape
        h = self.cpc.input_proj(x.view(-1, D)).view(B, T, -1)
        if self.has_abs_pos:
            h = self.cpc.pos_encoding(h)  # RoPE models won't have this attribute
        h = self.cpc.encoder(h)          # [B, T, H]
        #h = self.cpc.norm(h)
        #z = self.cpc.projector(h)        # [B, T, P]
        return h

    @torch.no_grad()
    def last_step(self, x: torch.Tensor):
        return self.forward(x)[:, -1, :]

    @torch.no_grad()
    def mean_pool(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.forward(x)  # [B, T, P]
        if mask is None:
            return z.mean(dim=1)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return (z * mask.unsqueeze(-1)).sum(dim=1) / denom

# ---------------------------------------------------------------------
# IO + preprocessing (aligned with training-time stats)
# ---------------------------------------------------------------------
# ---------- Preprocessing helpers (aligned to training) ----------
def load_train_stats(path: str):
    """
    Expect an .npz with keys: median, mean, std (shape [D]).
    """
    stats = np.load(path)
    med = stats['imputer_med'].astype(np.float32)
    mu  = stats['scaler_mu'  ].astype(np.float32)
    sd  = stats['scaler_sd'  ].astype(np.float32)
    return med, mu, sd

def load_patient_signal(file_path: str, feats: list[str]) -> np.ndarray:
    """
    Read CSV -> keep columns in the 'feats' order; missing columns become NaN.
    Return float32 array with NaNs preserved for bad values.
    """
    df = pd.read_csv(file_path, usecols=lambda c: c in feats)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.reindex(columns=feats)
    return df.to_numpy(dtype=np.float32)

def robust_stats_for_feature(col: np.ndarray, fallback=0.0):
    """
    Compute a robust median for a feature when the train median is NaN.
    """
    m = np.nanmedian(col)
    if not np.isfinite(m):
        m = fallback
    return m

def preprocess_window(X: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Impute NaNs with train medians per feature, then z-score with train mean/std.
    X: [T, D]
    """
    T, D = X.shape
    med_use = med.copy()
    bad_med = ~np.isfinite(med_use)
    if np.any(bad_med):
        for j in np.where(bad_med)[0]:
            med_use[j] = robust_stats_for_feature(X[:, j], fallback=0.0)

    mu_use = mu.copy()
    sd_use = sd.copy()
    sd_use[~np.isfinite(sd_use) | (sd_use == 0)] = 1.0
    mu_use[~np.isfinite(mu_use)] = 0.0

    med_row = np.broadcast_to(med_use, (T, D))
    X_imp = np.where(np.isfinite(X), X, med_row)
    X_z = (X_imp - mu_use) / sd_use
    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X_z

def make_window(X: np.ndarray, window_len: int) -> np.ndarray | None:
    """
    Take the first window_len samples; return None if too short.
    """
    if X.shape[0] < window_len:
        return None
    return X[:window_len, :]

# ---------------------------------------------------------------------
# Feature computation from a single window
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_stats_features(X: np.ndarray) -> np.ndarray:
    """
    X: [T, D] (already preprocessed)
    Returns [mean_j, var_j] concatenated -> shape [2D]
    """
    m = np.mean(X, axis=0)
    v = np.var(X, axis=0)
    return np.concatenate([m, v]).astype(np.float32)

@torch.no_grad()
def compute_window_feature_vector(
    feature_model: CPCFeatureExtractor,
    X_window: np.ndarray,           # [T, D] raw window (with NaNs)
    med: np.ndarray, mu: np.ndarray, sd: np.ndarray,
    mode: str = 'cpc_last'          # 'cpc_last' | 'cpc_mean' | 'stats' | 'concat'
) -> np.ndarray:
    """
    Produce a feature vector from a single window for prediction.
    """
    Xp = preprocess_window(X_window, med, mu, sd)

    if mode == 'stats':
        return compute_stats_features(Xp)

    x_tensor = torch.tensor(Xp).unsqueeze(0).to(device)  # [1, T, D]

    if mode == 'cpc_mean':
        z = feature_model.mean_pool(x_tensor)            # [1, P]
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    if mode == 'cpc_last':
        z = feature_model.last_step(x_tensor)            # [1, P]
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    if mode == 'concat':
        z_last = feature_model.mean_pool(x_tensor).squeeze(0).cpu().numpy()
        s_vec  = compute_stats_features(Xp)
        return np.concatenate([z_last.astype(np.float32), s_vec])

    raise ValueError(f"Unknown feature mode: {mode}")

def build_feature_columns(mode: str, feats: list[str], feature_dim: int, proj_dim_hint: int | None = None):
    """
    Create human-readable column names for the DataFrame.
    - For 'stats': returns <feat>_mean and <feat>_var
    - For 'cpc_*': returns cpc_0..cpc_{P-1}
    - For 'concat': cpc_* then stats
    """
    D = len(feats)
    if mode == 'stats':
        cols = []
        for f in feats:
            cols.append(f"{f}_mean")
            cols.append(f"{f}_var")
        return cols

    if mode in ('cpc_last', 'cpc_mean'):
        P = feature_dim
        return [f"cpc_{i}" for i in range(P)]

    if mode == 'concat':
        # feature_dim = P + 2D  -> infer P
        P = feature_dim - 2 * D
        cpc_cols = [f"cpc_{i}" for i in range(max(P, 0))]
        stat_cols = []
        for f in feats:
            stat_cols.append(f"{f}_mean")
            stat_cols.append(f"{f}_var")
        return cpc_cols + stat_cols

    # Fallback
    return [f"feat_{i}" for i in range(feature_dim)]

# ---------------------------------------------------------------------
# Batch extraction over files
# ---------------------------------------------------------------------
@torch.no_grad()
def extract_features_for_paths(
    feature_model: CPCFeatureExtractor,
    csv_paths: list[str],
    feats: list[str],
    window_len: int,
    med: np.ndarray, mu: np.ndarray, sd: np.ndarray,
    mode: str = 'cpc_last'
):
    """
    Iterate files, take a fixed window and export features (mode controls type).
    """
    all_features, ids = [], []
    for path in tqdm(csv_paths, desc=f"Extracting features ({mode})"):
        X = load_patient_signal(path, feats)      # [T, D] with NaNs
        Xw = make_window(X, window_len)
        if Xw is None:
            continue
        fv = compute_window_feature_vector(feature_model, Xw, med, mu, sd, mode=mode)
        all_features.append(fv)
        ids.append(os.path.basename(path).replace(".csv", ""))
    return np.stack(all_features), ids

# ------------------- Streaming (chunked) full-recording stats -------------------
def _interleave_mean_var(means: np.ndarray, vars_: np.ndarray, feats: list[str]) -> np.ndarray:
    out = np.empty(len(feats) * 2, dtype=np.float32)
    out[0::2] = means.astype(np.float32)
    out[1::2] = vars_.astype(np.float32)
    return out

def compute_full_recording_stats_streaming(file_path: str, feats: list[str], chunksize: int = 200_000) -> np.ndarray:
    """
    Stream a CSV and compute per-feature mean/variance across ALL rows (NaN-aware).
    Missing columns are treated as all-NaN. Returns [f1_mean, f1_var, f2_mean, f2_var, ...].
    """
    D = len(feats)
    sums   = np.zeros(D, dtype=np.float64)
    sumsq  = np.zeros(D, dtype=np.float64)
    counts = np.zeros(D, dtype=np.int64)

    for chunk in pd.read_csv(file_path, usecols=lambda c: c in feats, chunksize=chunksize):
        for c in chunk.columns:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.reindex(columns=feats)

        arr = chunk.to_numpy(dtype=np.float32, copy=False)   # [N, D]
        mask = np.isfinite(arr)

        counts += mask.sum(axis=0)
        safe = np.where(mask, arr, 0.0)
        sums  += safe.sum(axis=0, dtype=np.float64)
        sumsq += np.square(safe, dtype=np.float64).sum(axis=0, dtype=np.float64)

    means = np.full(D, np.nan, dtype=np.float64)
    vars_ = np.full(D, np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    vars_[nz] = (sumsq[nz] / counts[nz]) - np.square(means[nz])
    vars_[nz] = np.maximum(vars_[nz], 0.0)  # numeric safety

    return _interleave_mean_var(means, vars_, feats)

def extract_full_stats_for_paths_streaming(csv_paths: list[str], feats: list[str], chunksize: int = 200_000):
    """
    Streaming stats for each file. One feature vector per CSV.
    """
    all_features, ids = [], []
    for path in tqdm(csv_paths, desc="Extracting full-recording stats (streaming)"):
        fv = compute_full_recording_stats_streaming(path, feats, chunksize=chunksize)
        all_features.append(fv)
        ids.append(os.path.basename(path).replace(".csv", ""))
    return np.stack(all_features), ids


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------
    from model import CPCModel

    WINDOW_SIZE = int(60 * 60 / 20)   # 180 samples if features are every 20s over 1 hour
    D_INPUT = len(PPG_FEATURE_BASES)
    USE_STREAMING_FULL_STATS = False # <— turn streaming on/off
    FEAT_MODE = 'cpc_mean'  # keep 'stats' for full-recording stats path

    # --------------------------------------------------------------
    # Load CPC backbone + weights, then wrap as feature extractor
    # --------------------------------------------------------------
    cpc = CPCModel(
        input_dim=D_INPUT,
        hidden_dim=256,
        proj_dim=128,
        heads=8,
        depth=4,
        max_seq_len=WINDOW_SIZE,
        pred_steps=1
    ).to(device)

    ckpt = torch.load("cpc_best.pt", map_location=device)
    cpc.load_state_dict(ckpt, strict=True)
    feature_model = CPCFeatureExtractor(cpc).to(device).eval()

    # --------------------------------------------------------------
    # Load train-time preprocessing stats (median/mean/std per feature)
    # --------------------------------------------------------------
    med, mu, sd = load_train_stats("train_stats_ppg.npz")

    # --------------------------------------------------------------
    # Input data
    # --------------------------------------------------------------
    csv_paths = sorted(glob.glob("../data/PPG/*.csv"))

    # --------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------
    if FEAT_MODE == 'stats' and USE_STREAMING_FULL_STATS:
        # Streaming, NaN-aware mean/var over the entire recording
        X, ids = extract_full_stats_for_paths_streaming(csv_paths, PPG_FEATURE_BASES, chunksize=200_000)
        columns = [f"{f}_{w}" for f in PPG_FEATURE_BASES for w in ("mean", "var")]
    else:
        # Your existing windowed path (unchanged)
        X, ids = extract_features_for_paths(
            feature_model,
            csv_paths,
            PPG_FEATURE_BASES,
            window_len=WINDOW_SIZE,
            med=med, mu=mu, sd=sd,
            mode=FEAT_MODE
        )
        columns = build_feature_columns(FEAT_MODE, PPG_FEATURE_BASES, feature_dim=X.shape[1])


    features_df = pd.DataFrame(X, index=ids, columns=columns)
    # Normalize IDs (keep subject part only)
    features_df.index = features_df.index.str.replace(r'^.*?-', '', regex=True)  # remove prefix like '01NVa-'
    features_df.index = features_df.index.str.replace(r'_.*$', '', regex=True)  # remove suffix like '_20s'
    features_df.index.name = 'USUBJID'

    # --------------------------------------------------------------
    # Load labels
    # --------------------------------------------------------------
    print("Loading labels...")
    excel_data = pd.ExcelFile('../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx')
    labels = pd.read_excel(excel_data, 'CLI')[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
    labels = labels.astype({col: int for col in ['DIED', 'HFNOV', 'MEVEN']})
    labels = labels.set_index('USUBJID')

    # --------------------------------------------------------------
    # Join features and labels
    # --------------------------------------------------------------
    dataset = features_df.join(labels, how='inner')

    # Optional: remove specific IDs for DIED label only
    label_name = 'DIED'
    if label_name == 'DIED':
        remove_ids = ['003-1011', '003-1016', '003-1017', '003-1214', '003-1219', '003-1239']
        dataset = dataset[~dataset.index.isin(remove_ids)]

    # --------------------------------------------------------------
    # Prepare final arrays
    # --------------------------------------------------------------
    X = dataset[columns].values
    y = dataset[label_name].values

    # --------------------------------------------------------------
    # Repeated Stratified K-Fold with RF + train-derived threshold
    # --------------------------------------------------------------
    n_repeats, n_splits = 20, 10

    aucs, auprcs, accs, sens_list, spec_list, f1s = [], [], [], [], [], []

    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # safety net for any NaNs
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(random_state=42))
            ])
            clf.fit(X_train, y_train)

            # Youden J threshold from training set
            y_prob_train = clf.predict_proba(X_train)[:, 1]
            fpr, tpr, thr = roc_curve(y_train, y_prob_train)
            opt_thr = thr[(tpr - fpr).argmax()]

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= opt_thr).astype(int)

            y_true_all.extend(y_test.tolist())
            y_prob_all.extend(y_prob.tolist())
            y_pred_all.extend(y_pred.tolist())

        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        y_pred_all = np.array(y_pred_all)

        auc  = roc_auc_score(y_true_all, y_prob_all)
        auprc= average_precision_score(y_true_all, y_prob_all)
        acc  = (y_pred_all == y_true_all).mean()
        sens = recall_score(y_true_all, y_pred_all, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = f1_score(y_true_all, y_pred_all, zero_division=0)

        aucs.append(auc); auprcs.append(auprc); accs.append(acc)
        sens_list.append(sens); spec_list.append(spec); f1s.append(f1)

        print(f"Repeat {repeat + 1}/{n_repeats} — "
              f"AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | "
              f"Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    print(f"\n=== Repeated {n_splits}-Fold CV (mean ± std over {n_repeats} repeats) ===")
    print(f"AUROC    : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AUPRC    : {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Sensitivity (TPR): {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}")
    print(f"Specificity (TNR): {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
    print(f"F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
