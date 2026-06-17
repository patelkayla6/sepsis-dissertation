import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# -----------------------------
#  SAME PPG FEATURE LIST
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


class VietnamPPGDataset(Dataset):
    def __init__(self, vn_pattern, max_files=None):
        X = load_all_rows_features(vn_pattern, max_files=max_files)
        # You *could* standardize with MIMIC stats; here we just standardize per-Vietnam
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - self.mean) / self.std
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# ----- same VQ + model as training -----
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
        B, D = z.shape
        z_flat = z.view(-1, D)
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        encoding_indices = d.argmin(1)
        encodings = F.one_hot(encoding_indices, self.n_code).type(z.dtype)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view_as(z)

        if self.training:
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

        quantized_st = z + (quantized - z).detach()
        vq_loss = F.mse_loss(quantized.detach(), z)
        commit_loss = F.mse_loss(quantized, z.detach())
        return quantized_st, vq_loss, commit_loss, encoding_indices.view(B)


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
        z_e = self.encoder(x)
        z_q, vq_loss, commit_loss, idx = self.quantizer(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, commit_loss, idx


# -----------------------------
#  ANALYSE VIETNAM DISTRIBUTION
# -----------------------------
def analyse_vietnam_distribution(
    ckpt_path,
    vn_pattern,
    n_code,
    hidden_dim,
    code_dim,
    device="cuda",
    batch_size=64,
    max_files=None,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Build Vietnam dataset & loader
    vn_dataset = VietnamPPGDataset(vn_pattern, max_files=max_files)
    input_dim = vn_dataset.X.shape[1]
    vn_loader = DataLoader(vn_dataset, batch_size=batch_size, shuffle=False)

    # 2. Build model with same dims and load checkpoint
    model = VQFeatureAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        n_code=n_code,
        ema_decay=0.99,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Collect code indices
    all_idx = []
    with torch.no_grad():
        for x in tqdm(vn_loader, desc="Encoding Vietnam data"):
            x = x.to(device)
            z_e = model.encoder(x)
            # call quantizer, but EMA won't update because model.eval()
            _, _, _, idx = model.quantizer(z_e)
            all_idx.append(idx.cpu().numpy())

    all_idx = np.concatenate(all_idx)  # shape (N,)
    print("Total encoded tokens:", all_idx.shape[0])

    # 4. Histogram
    counts = np.bincount(all_idx, minlength=n_code)
    probs = counts / counts.sum()

    print("Non-empty codes:", np.sum(counts > 0), "/", n_code)
    print("Top-10 most used codes (code_id, count):")
    top10 = np.argsort(counts)[::-1][:10]
    for cid in top10:
        print(cid, int(counts[cid]))

    # 5. Plot
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(n_code), probs)
    plt.xlabel("Code index")
    plt.ylabel("Frequency (Vietnam)")
    plt.title("Codebook usage distribution on Vietnam PPG features")
    plt.tight_layout()
    plt.show()

    return counts, probs


if __name__ == "__main__":
    # ---- EDIT THESE PATHS / PARAMS ----
    CKPT_PATH = "vq_ppg_ckpt/epoch_100.pt"   # your trained MI checkpoint
    VN_PATTERN = "../../data/PPG/*.csv"      # your Vietnam PPG feature files
    N_CODE = 8192
    HIDDEN_DIM = 256
    CODE_DIM = 128

    analyse_vietnam_distribution(
        ckpt_path=CKPT_PATH,
        vn_pattern=VN_PATTERN,
        n_code=N_CODE,
        hidden_dim=HIDDEN_DIM,
        code_dim=CODE_DIM,
        device="cuda",
        batch_size=64,
        max_files=None,   # or e.g. 160
    )
