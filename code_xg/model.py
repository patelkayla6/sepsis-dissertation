#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from x_transformers import Decoder

import matplotlib.pyplot as plt

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Feature bases (PPG)
# ---------------------------
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

D_INPUT = len(PPG_FEATURE_BASES)


# ---------------------------
# IO helpers
# ---------------------------
def load_patient_signal(csv_path: str, feats: list[str]) -> np.ndarray:
    """
    Read CSV, keep the full feature schema order.
    - Any missing columns → filled with NaN (kept as NaN here).
    - Values parsed to numeric; inf → NaN.
    - No imputation or normalization here.
    Returns: [T, D] float32 (with NaNs).
    """
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Build in the canonical order with NaNs if missing
    cols = {}
    for f in feats:
        if f in df.columns:
            col = pd.to_numeric(df[f], errors='coerce')
        else:
            # column entirely missing → NaNs with same length as any existing column or 0 if empty file
            L = len(df) if len(df) > 0 else 0
            col = pd.Series([np.nan] * L, dtype="float64")
        cols[f] = col

    out = pd.DataFrame(cols)[feats].to_numpy(dtype=np.float32)
    return out


# ---------------------------
# HDF5 curation (sliding windows)
# ---------------------------
def curate_hdf5_windows(csv_paths: list[str], h5_path: str, feats: list[str], window_size: int, step: int = 1):
    """
    Save per-patient sliding windows to HDF5 as /patient_<idx>: [num_windows, T, D]
    Skips very short signals (< window_size+1).
    """
    with h5py.File(h5_path, 'w') as h5f:
        for pid, csv_path in enumerate(tqdm(csv_paths, desc="Curating HDF5")):
            try:
                x = load_patient_signal(csv_path, feats=feats)  # [T, D] with NaNs
                T = x.shape[0]
                if T < window_size + 1:
                    continue
                # build windows
                num_windows = (T - window_size)
                idxs = range(0, num_windows, step)
                windows = np.stack([x[i:i + window_size] for i in idxs], axis=0)  # [N, W, D]
                h5f.create_dataset(f'patient_{pid}', data=windows) #, compression="gzip", compression_opts=3)
            except Exception as e:
                print(f"Skipping {csv_path} due to error: {e}")


# ---------------------------
# Train-only stats from HDF5
# ---------------------------
def compute_train_stats_from_h5(h5_path: str, val_split: float = 0.2):
    import h5py, numpy as np
    with h5py.File(h5_path, 'r') as h5:
        vals_for_median = []
        sum_, sumsq_, count_ = None, None, None

        for pid in tqdm(h5):
            arr = h5[pid]  # [N, W, D]
            N = arr.shape[0]
            split_idx = max(int(N * (1 - val_split)) - 1, 0)
            if split_idx <= 0:
                continue
            train = arr[:split_idx]                       # [N_train, W, D]
            flat = train.reshape(-1, train.shape[-1])     # [M, D] (may contain NaN)

            vals_for_median.append(flat)

            valid = np.isfinite(flat)
            flat_nan0 = np.where(valid, flat, 0.0)

            col_sum   = flat_nan0.sum(axis=0)
            col_sumsq = (flat_nan0 * flat_nan0).sum(axis=0)
            col_count = valid.sum(axis=0)  # number of finite values per feature

            if sum_ is None:
                sum_, sumsq_, count_ = col_sum, col_sumsq, col_count
            else:
                sum_   += col_sum
                sumsq_ += col_sumsq
                count_ += col_count

        if not vals_for_median:
            raise RuntimeError("No training windows found to compute statistics.")

        flat_all = np.vstack(vals_for_median)  # [Total_M, D]
        medians = np.nanmedian(flat_all, axis=0)  # NaN if a column is entirely NaN

        # mean/std from sums
        means = np.divide(sum_, np.maximum(count_, 1), where=(np.maximum(count_, 1) > 0))
        vars_ = np.divide(sumsq_, np.maximum(count_, 1), where=(np.maximum(count_, 1) > 0)) - (means ** 2)
        vars_ = np.maximum(vars_, 0.0)
        stds  = np.sqrt(vars_)

        # Identify features with zero finite samples in train
        zero_mask = (count_ == 0)

        # Fallbacks for dead features (safe, non-informative)
        medians[zero_mask] = 0.0
        means[zero_mask]   = 0.0
        stds[zero_mask]    = 1.0

        # Final guards
        means[~np.isfinite(means)] = 0.0
        stds[(~np.isfinite(stds)) | (stds == 0)] = 1.0
        medians[~np.isfinite(medians)] = 0.0

        return medians.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


# ---------------------------
# Dataset (applies train stats inside)
# ---------------------------
class HDF5HRVDataset(Dataset):
    def __init__(self, h5_path: str, mode: str = 'train', val_split: float = 0.2,
                 imputer_median: np.ndarray | None = None,
                 scaler_mean: np.ndarray | None = None,
                 scaler_std: np.ndarray | None = None):
        assert mode in ('train', 'val')
        self.h5 = h5py.File(h5_path, 'r')
        self.mode = mode
        self.val_split = val_split
        self.imputer_median = imputer_median
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std

        self.index_map: list[tuple[str, int]] = []
        for pid in self.h5:
            N = self.h5[pid].shape[0]
            split_idx = max(int(N * (1 - val_split)) - 1, 0)
            if mode == 'train':
                # windows [0 .. split_idx-1] are train, and we predict next window's last step in SSL variants
                self.index_map += [(pid, i) for i in range(max(split_idx - 1, 0))]
            else:
                self.index_map += [(pid, i) for i in range(max(split_idx - 1, 0), N - 1)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        pid, i = self.index_map[idx]
        x = self.h5[pid][i]        # [W, D]
        x_next = self.h5[pid][i+1] # [W, D] (used in some SSL targets)

        # median impute (train-only stats)
        if self.imputer_median is not None:
            med = self.imputer_median[None, :]  # [1, D]
            nan_mask = np.isnan(x)
            if nan_mask.any():
                x = np.where(nan_mask, med, x)
            nan_mask2 = np.isnan(x_next)
            if nan_mask2.any():
                x_next = np.where(nan_mask2, med, x_next)

        # z-score normalize (train-only stats)
        if (self.scaler_mean is not None) and (self.scaler_std is not None):
            mu = self.scaler_mean[None, :]
            sd = self.scaler_std[None, :]
            x = (x - mu) / (sd + 1e-8)
            x_next = (x_next - mu) / (sd + 1e-8)


        # Return both for SSL
        x = torch.tensor(x, dtype=torch.float32)
        x_next = torch.tensor(x_next, dtype=torch.float32)
        return x, x_next


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ---------------------------
# CPC-style SSL model
# ---------------------------
class CPCModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, proj_dim: int = 128,
                 heads: int = 8, depth: int = 4, max_seq_len: int = 360, pred_steps: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.encoder = Decoder(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            rotary_xpos = True,               # <<— turn on RoPE
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(hidden_dim, proj_dim)
        self.predictors = nn.ModuleList([nn.Linear(proj_dim, proj_dim) for _ in range(pred_steps)])
        self.pred_steps = pred_steps

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        B, T, D = x.shape
        h = self.input_proj(x.reshape(-1, D)).reshape(B, T, -1)
        #h = self.pos_encoding(h)
        z = self.encoder(h)           # [B, T, H]
        z = self.norm(z)
        z = self.projector(z)         # [B, T, P]

        # context excludes last pred_steps to allow predicting future positions
        context = z[:, :-self.pred_steps, :]  # [B, T - K, P]
        preds = []
        targets = []
        for k in range(self.pred_steps):
            preds.append(self.predictors[k](context))           # [B, T-K, P]
            targets.append(z[:, k+1: z.size(1)-self.pred_steps+k+1, :])  # aligned future slice
        return preds, targets


def cpc_loss_from_lists(preds, targets, temperature: float = 0.07) -> torch.Tensor:
    """
    preds_list / targets_list: list of tensors with shape [B, T, P]
    InfoNCE over the batch-time axis for each k, averaged across k.
    """

    B, T, P = preds.shape
    preds = preds.reshape(B * T, P)
    targets = targets.reshape(B * T, P)
    preds = F.normalize(preds, dim=-1)
    targets = F.normalize(targets, dim=-1)
    logits = preds @ targets.t() / temperature
    labels = torch.arange(B * T, device=preds.device)
    loss = F.cross_entropy(logits, labels)

    return loss


# ---------------------------
# Training loops
# ---------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0.; self.sum = 0.; self.count = 0; self.avg = 0.
    def update(self, v, n=1): self.val = v; self.sum += v * n; self.count += n; self.avg = self.sum / max(self.count, 1)


def train_cpc(model, train_loader, val_loader, epochs=100, lr=1e-3, grad_clip=None, ckpt_path="cpc_best.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_sum, train_count = 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch") as pbar:
            for x, _ in pbar:
                x = x.to(device)
                optimizer.zero_grad()
                preds_list, targets_list = model(x)

                # merge CPC loss + MSE loss
                loss_total = 0.0

                for preds, targets in zip(preds_list, targets_list):
                    loss_total += F.mse_loss(preds, targets)

                loss_total.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_sum += loss_total.item() * x.size(0)
                train_count += x.size(0)
                pbar.set_postfix(avg_loss=train_sum / train_count)

        # ---- Validation ----
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", unit="batch") as pbar:
                for x, _ in pbar:
                    x = x.to(device)
                    preds_list, targets_list = model(x)
                    loss_total = 0.0
                    for preds, targets in zip(preds_list, targets_list):
                        loss_total += F.mse_loss(preds, targets)

                    val_sum += loss_total.item() * x.size(0)
                    val_count += x.size(0)
                    pbar.set_postfix(avg_loss=val_sum / val_count)

        avg_val_loss = val_sum / val_count
        print(f"Epoch {epoch} Summary: Train Avg Loss={train_sum/train_count:.4f} | Val Avg Loss={avg_val_loss:.4f}")

        # ---- Save Best ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ New best model saved at {ckpt_path} (Val Loss={best_val_loss:.4f})")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # --- Paths and basic config
    DATA_DIR = "../data/"
    PPG_DIR = os.path.join(DATA_DIR, "PPG")
    H5_PATH = "ppg_windows.h5"

    # --- Sliding window: 20-second samples → 1 hour window (example)
    # If each row is 20s summary, one hour = 180 samples.
    WINDOW_SIZE = int(60 * 60 / 20)  # 180
    STEP = 180

    # --- List CSVs
    csv_paths = sorted(glob.glob(os.path.join(PPG_DIR, "*.csv")))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSVs found in {PPG_DIR}")

    # --- Build HDF5 if not present
    if not os.path.exists(H5_PATH):
        curate_hdf5_windows(csv_paths, H5_PATH, feats=PPG_FEATURE_BASES,
                            window_size=WINDOW_SIZE, step=STEP)

    # --- Compute train-only stats (median / mean / std)
    '''
    this is extremley time-consuming
    '''

    STATS_FILE = "train_stats_ppg.npz"

    if not os.path.exists(STATS_FILE):
        imputer_med, scaler_mu, scaler_sd = compute_train_stats_from_h5(H5_PATH, val_split=0.2)
        np.savez(
            STATS_FILE,
            imputer_med=imputer_med,
            scaler_mu=scaler_mu,
            scaler_sd=scaler_sd
        )
        print(f"✅ Stats saved to {STATS_FILE}")
    else:
        data = np.load(STATS_FILE)
        imputer_med = data["imputer_med"]
        scaler_mu = data["scaler_mu"]
        scaler_sd = data["scaler_sd"]
        print(f"📂 Loaded stats from {STATS_FILE}")
    # imputer_med, scaler_mu, scaler_sd = 0.,0.,0.

    # --- Datasets / Loaders
    train_ds = HDF5HRVDataset(
        H5_PATH, mode='train', val_split=0.2,
        imputer_median=imputer_med, scaler_mean=scaler_mu, scaler_std=scaler_sd
    )
    val_ds = HDF5HRVDataset(
        H5_PATH, mode='val', val_split=0.2,
        imputer_median=imputer_med, scaler_mean=scaler_mu, scaler_std=scaler_sd
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model
    model = CPCModel(
        input_dim=D_INPUT,
        hidden_dim=128,
        proj_dim=128,
        heads=8,
        depth=4,
        max_seq_len=WINDOW_SIZE,
        pred_steps=1
    ).to(device)

    # --- Train
    train_cpc(model, train_loader, val_loader, epochs=100, lr=1e-3)

    # --- Save weights
    torch.save(model.state_dict(), "gpt_feature_model.pt")
    print("Saved model to gpt_feature_model.pt")
