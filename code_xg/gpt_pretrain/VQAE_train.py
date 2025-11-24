import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
#  PPG feature list (same as before)
# -----------------------------
PPG_FEATURE_NAMES = [
    'Tpi','Tpp','Tsys','Tdia','Tsp','Tdp','deltaT',
    'Tsw10','Tsw25','Tsw33','Tsw50','Tsw66','Tsw75','Tsw90',
    'Tdw10','Tdw25','Tdw33','Tdw50','Tdw66','Tdw75','Tdw90',
    'Tpw10','Tpw25','Tpw33','Tpw50','Tpw66','Tpw75','Tpw90',
    'Asp','Adn','Adp','Aoff','AUCpi','AUCsys','AUCdia','SQI',
    'HRV_MeanNN','HRV_SDNN','HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2',
    'HRV_SDANN5','HRV_SDNNI5','HRV_RMSSD','HRV_SDSD','HRV_CVNN','HRV_CVSD',
    'HRV_MedianNN','HRV_MadNN','HRV_MCVNN','HRV_IQRNN','HRV_SDRMSSD',
    'HRV_Prc20NN','HRV_Prc80NN','HRV_pNN50','HRV_pNN20','HRV_MinNN','HRV_MaxNN',
    'HRV_HTI','HRV_TINN','HRV_ULF','HRV_VLF','HRV_LF','HRV_HF','HRV_VHF',
    'HRV_TP','HRV_LFHF','HRV_LFn','HRV_HFn','HRV_LnHF',
    'HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S',
    'HRV_CSI','HRV_CVI','HRV_CSI_Modified','HRV_PIP','HRV_IALS',
    'HRV_PSS','HRV_PAS','HRV_GI','HRV_SI','HRV_AI','HRV_PI',
    'HRV_C1d','HRV_C1a','HRV_SD1d','HRV_SD1a','HRV_C2d','HRV_C2a',
    'HRV_SD2d','HRV_SD2a','HRV_Cd','HRV_Ca','HRV_SDNNd','HRV_SDNNa',
    'HRV_DFA_alpha1',
    'HRV_MFDFA_alpha1_Width','HRV_MFDFA_alpha1_Peak','HRV_MFDFA_alpha1_Mean',
    'HRV_MFDFA_alpha1_Max','HRV_MFDFA_alpha1_Delta','HRV_MFDFA_alpha1_Asymmetry',
    'HRV_MFDFA_alpha1_Fluctuation','HRV_MFDFA_alpha1_Increment',
    'HRV_ApEn','HRV_SampEn','HRV_ShanEn','HRV_FuzzyEn',
    'HRV_MSEn','HRV_CMSEn','HRV_RCMSEn',
    'HRV_CD','HRV_HFD','HRV_KFD','HRV_LZC',
]


# -----------------------------
#  Dataset: first-row PPG features per file
# -----------------------------
def load_first_row_features(pattern, max_files=None):
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    vectors = []

    for fp in tqdm(files, desc=f"Loading {pattern}"):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp} — read error: {e}")
            continue

        row = []
        for col in PPG_FEATURE_NAMES:
            if col in df.columns:
                try:
                    row.append(float(df[col].iloc[0]))
                except Exception:
                    row.append(0.0)
            else:
                row.append(0.0)

        vectors.append(row)

    X = np.array(vectors, dtype=np.float32)
    # clean: replace NaN/inf and clip extreme values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X

def load_all_rows_features(pattern, max_files=None, max_rows_per_file=None):
    """
    Load ALL rows from each CSV matching `pattern`.
    Each row -> one feature vector (len(PPG_FEATURE_NAMES)).

    Args:
        pattern: glob pattern for csv files.
        max_files: if not None, limit number of files.
        max_rows_per_file: if not None, take at most this many rows per file.

    Returns:
        X: np.ndarray of shape (N_rows_total, len(PPG_FEATURE_NAMES))
    """
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    vectors = []

    for fp in tqdm(files, desc=f"Loading {pattern} (all rows)"):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp} — read error: {e}")
            continue

        n_rows = len(df)
        if max_rows_per_file is not None:
            n_rows = min(n_rows, max_rows_per_file)

        for i in range(n_rows):
            row_vec = []
            for col in PPG_FEATURE_NAMES:
                if col in df.columns:
                    try:
                        row_vec.append(float(df[col].iloc[i]))
                    except Exception:
                        row_vec.append(0.0)
                else:
                    row_vec.append(0.0)
            vectors.append(row_vec)

    X = np.array(vectors, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X


class PPGFeatureDataset(Dataset):
    def __init__(self, mi_pattern, max_per_site=None):
        #self.X = load_first_row_features(mi_pattern, max_files=max_per_site)
        self.X = load_all_rows_features(mi_pattern, max_files=max_per_site)

        # standardize
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std

        self.X = torch.from_numpy(self.X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# -----------------------------
#  EMA Vector Quantizer
# -----------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_code, code_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_code = n_code
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(n_code, code_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer("ema_cluster_size", torch.zeros(n_code))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z):
        # z: (B, D)
        B, D = z.shape
        z_flat = z.view(-1, D)

        # distances to codebook
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )  # (B, n_code)

        encoding_indices = d.argmin(1)  # (B,)
        encodings = F.one_hot(encoding_indices, self.n_code).type(z.dtype)

        # quantize
        quantized = encodings @ self.embedding.weight  # (B, D)
        quantized = quantized.view_as(z)

        if self.training:
            # EMA updates
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            dw = encodings.t() @ z_flat
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.n_code * self.eps)
                * n
            )
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # VQ loss
        vq_loss = F.mse_loss(quantized.detach(), z)
        commit_loss = F.mse_loss(quantized, z.detach())
        return quantized_st, vq_loss, commit_loss, encoding_indices.view(B)


# -----------------------------
#  Simple VQ-AE model for features
# -----------------------------
class VQFeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, n_code, ema_decay=0.99):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, code_dim),
        )
        self.quantizer = VectorQuantizerEMA(n_code, code_dim, decay=ema_decay)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z_e = self.encoder(x)             # (B, code_dim)
        z_q, vq_loss, commit_loss, idx = self.quantizer(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, commit_loss, idx


# -----------------------------
#  Training loop
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--vn_pattern", type=str,
    #                default="../../data/PPG/*.csv")
    p.add_argument("--mi_pattern", type=str,
                   default="../../code_xg/preprocess_MIMIC-III/ppg_features_by_icu_v2/*.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--code_dim", type=int, default=128)
    p.add_argument("--n_code", type=int, default=8192)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="vq_ppg_ckpt")
    p.add_argument("--max_per_site", type=int, default=None)  # e.g. 160
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PPGFeatureDataset(
        # vn_pattern=args.vn_pattern,
        mi_pattern=args.mi_pattern,
        max_per_site=args.max_per_site,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)

    input_dim = dataset.X.shape[1]
    model = VQFeatureAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        n_code=args.n_code,
        ema_decay=args.ema_decay,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"Start training: N={len(dataset)}, input_dim={input_dim}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_vq = 0.0
        total_commit = 0.0

        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()

            x_rec, vq_loss, commit_loss, _ = model(x)
            rec_loss = F.mse_loss(x_rec, x)
            loss = rec_loss + vq_loss + 0.25 * commit_loss  # beta=0.25

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_rec += rec_loss.item() * x.size(0)
            total_vq += vq_loss.item() * x.size(0)
            total_commit += commit_loss.item() * x.size(0)

        n = len(dataset)
        print(
            f"Epoch {epoch:03d}: "
            f"loss={total_loss/n:.4f}, rec={total_rec/n:.4f}, "
            f"vq={total_vq/n:.4f}, commit={total_commit/n:.4f}"
        )

        # save checkpoint + codebook
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = Path(args.output_dir) / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            # also save the codebook itself
            codebook = model.quantizer.embedding.weight.detach().cpu().numpy()
            np.save(Path(args.output_dir) / f"codebook_epoch_{epoch:03d}.npy", codebook)


if __name__ == "__main__":
    main()
