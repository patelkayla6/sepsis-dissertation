import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from x_transformers import TransformerWrapper, AutoregressiveWrapper, Decoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Feature Bases ---
PPG_feature_bases = [
    'Adn', 'Adp', 'Aoff', 'Asp', 'AUCdia', 'AUCpi', 'AUCsys',
    'Tdia', 'Tdp', 'Tpi', 'Tpp', 'Tsp', 'Tsys', 'deltaT',
    'Tsw10', 'Tsw25', 'Tsw33', 'Tsw50', 'Tsw66', 'Tsw75', 'Tsw90',
    'Tdw10', 'Tdw25', 'Tdw33', 'Tdw50', 'Tdw66', 'Tdw75', 'Tdw90',
    'Tpw10', 'Tpw25', 'Tpw33', 'Tpw50', 'Tpw66', 'Tpw75', 'Tpw90',
    'HRV_AI', 'HRV_ApEn', 'HRV_C1a', 'HRV_C1d', 'HRV_C2a',
    'HRV_C2d', 'HRV_CD', 'HRV_CSI', 'HRV_CSI_Modified', 'HRV_CVNN', 'HRV_CVI',
    'HRV_CVSD', 'HRV_Ca', 'HRV_Cd', 'HRV_DFA_alpha1', 'HRV_FuzzyEn',
    'HRV_GI', 'HRV_HFD', 'HRV_HF', 'HRV_HTI', 'HRV_IALS', 'HRV_IQRNN',
    'HRV_KFD', 'HRV_LZC', 'HRV_LnHF', 'HRV_MCVNN', 'HRV_MFDFA_alpha1_Asymmetry',
    'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
    'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Width', 'HRV_MadNN', 'HRV_MaxNN', 'HRV_MeanNN',
    'HRV_MedianNN', 'HRV_MinNN', 'HRV_PAS', 'HRV_PIP', 'HRV_PI', 'HRV_Prc20NN',
    'HRV_Prc80NN', 'HRV_PSS', 'HRV_SD1', 'HRV_SD1SD2', 'HRV_SD1a',
    'HRV_SD1d', 'HRV_SD2', 'HRV_SD2a', 'HRV_SD2d', 'HRV_SDNN',
    'HRV_SDNNd', 'HRV_SDNNa', 'HRV_SDRMSSD', 'HRV_SDSD', 'HRV_SI', 'HRV_S',
    'HRV_SampEn', 'HRV_ShanEn', 'HRV_TINN', 'HRV_TP', 'HRV_VHF', 'HRV_HFn'
]

# --- Data Loading & Imputation ---
def load_patient_signal(file_path, feats):
    df = pd.read_csv(file_path, usecols=feats)
    df = df.fillna(method='ffill').fillna(df.mean())
    for col in df.columns:
        m, s = df[col].mean(), df[col].std()
        df[col] = (df[col] - m) / (s + 1e-6)
        df[col] = df[col].fillna(0.0)
    return df.to_numpy(dtype=np.float32)

# --- Curate HDF5 with Per-Patient Temporal Split ---
def curate_hdf5_windows(patient_csv_paths, save_path, feats, window_size):
    with h5py.File(save_path, 'w') as h5f:
        for pid, csv_path in enumerate(patient_csv_paths):
            try:
                signal = load_patient_signal(csv_path, feats=feats)
                windows = []
                for i in range(len(signal) - window_size):
                    windows.append(signal[i:i + window_size])
                windows = np.stack(windows)
                h5f.create_dataset(f'patient_{pid}', data=windows)
            except Exception as e:
                print(f"Skipping {csv_path} due to error: {e}")

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Dataset ---
class HDF5HRVDataset(Dataset):
    def __init__(self, h5_path, mode='train', val_split=0.2):
        self.h5 = h5py.File(h5_path, 'r')
        self.index_map = []
        for pid in self.h5:
            L = self.h5[pid].shape[0]
            split_idx = int(L * (1 - val_split))
            if mode == 'train':
                self.index_map += [(pid, i) for i in range(split_idx - 1)]
            elif mode == 'val':
                self.index_map += [(pid, i) for i in range(split_idx - 1, L - 1)]

    def __len__(self): return len(self.index_map)

    def __getitem__(self, idx):
        pid, i = self.index_map[idx]
        x = self.h5[pid][i]
        y = self.h5[pid][i+1][-1]
        return torch.tensor(x), torch.tensor(y)

# --- Model ---
class GPTFeatureModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, proj_dim=128, heads=8, depth=4, max_seq_len=360):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.gpt = Decoder(dim=hidden_dim, depth=depth, heads=heads,
                           layer_dropout=0.1, attn_dropout=0.1, ff_dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.to_recon = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_in = x[:, :-1, :]
        x_target = x[:, 1:, :]
        h = self.input_proj(x_in.reshape(-1, D)).reshape(B, T-1, -1)
        h = self.pos_encoding(h)
        h = self.gpt(h)
        h = self.norm(h)
        z = self.to_latent(h)
        y_pred = self.to_recon(z)
        return y_pred, x_target

# --- Training ---
def train_autoregressive(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, _ = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred, y_true = model(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x, _ = x.to(device), y.to(device)
                y_pred, y_true = model(x)
                loss = loss_fn(y_pred, y_true)
                total_val_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))
        val_losses.append(total_val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

# --- Main ---
if __name__ == '__main__':
    h5_path = 'ppg_windows.h5'
    window_size = int(1 * 60 * 60 / 20)
    csv_paths = sorted(glob.glob("../data/PPG/*.csv"))

    if not os.path.exists(h5_path):
        curate_hdf5_windows(csv_paths, h5_path, feats=PPG_feature_bases, window_size=window_size)

    train_dataset = HDF5HRVDataset(h5_path, mode='train')
    val_dataset = HDF5HRVDataset(h5_path, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

    model = GPTFeatureModel(input_dim=len(PPG_feature_bases), max_seq_len=window_size).to(device)
    train_autoregressive(model, train_loader, val_loader, epochs=100)
    torch.save(model.state_dict(), "gpt_feature_model.pt")

