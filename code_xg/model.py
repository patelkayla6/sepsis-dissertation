import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def average(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def reset(self):
        self.sum = 0.0
        self.count = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

def load_patient_signal(file_path, feats):
    df = pd.read_csv(file_path, usecols=feats)
    df = df.fillna(method='ffill').fillna(df.mean())
    for col in df.columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[col] = (df[col] - col_mean) / (col_std + 1e-6)
        df[col] = df[col].fillna(0.0)
    return df.to_numpy(dtype=np.float32)

def curate_hdf5_windows(patient_csv_paths, save_path, feats, window_size):
    with h5py.File(save_path, 'w') as h5f:
        for pid, csv_path in enumerate(patient_csv_paths):
            try:
                print(csv_path)
                signal = load_patient_signal(csv_path, feats=feats)
                windows = []
                for i in range(len(signal) - window_size):
                    windows.append(signal[i:i + window_size])
                windows = np.stack(windows)
                h5f.create_dataset(f'patient_{pid}', data=windows, compression='gzip')
            except Exception as e:
                print(f"Skipping {csv_path} due to error: {e}")

class HDF5HRVDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.index_map = []
        for pid in self.h5_file:
            num_windows = self.h5_file[pid].shape[0]
            for i in range(num_windows - 1):
                self.index_map.append((pid, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        pid, i = self.index_map[idx]
        x = self.h5_file[pid][i]
        y = self.h5_file[pid][i + 1][0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class GRUFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, projection_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.reconstruction = nn.Linear(projection_dim, input_dim)

    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        z = self.projection(h)
        y_pred = self.reconstruction(z)
        return y_pred, z

class TransformerFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, projection_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.reconstruction = nn.Linear(projection_dim, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        h = x[:, -1, :]
        z = self.projection(h)
        y_pred = self.reconstruction(z)
        return y_pred, z


def train_autoregressive(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        running_avg = RunningAverage()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_avg.update(loss.item())
            progress_bar.set_postfix(loss=running_avg.average())
        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    h5_path = 'ppg_windows.h5'
    window_size = int(1 * 60 * 60 / 20)
    csv_paths = sorted(glob.glob("../data/PPG/*.csv"))

    if not os.path.exists(h5_path):
        print("Curating HDF5 file from raw CSVs...")
        curate_hdf5_windows(csv_paths, h5_path, feats=PPG_feature_bases, window_size=window_size)
    else:
        print(f"Found existing {h5_path}, skipping curation.")

    print(f"Loading data")
    dataset = HDF5HRVDataset(h5_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)

    print("Loading model")
    model = TransformerFeatureEncoder(input_dim=len(PPG_feature_bases), hidden_dim=128, projection_dim=128).to(device)
    train_autoregressive(model, loader, epochs=100)

