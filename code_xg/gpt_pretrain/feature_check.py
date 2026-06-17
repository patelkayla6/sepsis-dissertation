import os
import glob
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from tqdm import tqdm

PPG_feature_names = [
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

def load_first_row_features(pattern, site_label):
    files = sorted(glob.glob(pattern))[:]
    vectors = []
    labels = []

    for fp in tqdm(files):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp} — read error: {e}")
            continue

        row = []
        for col in PPG_feature_names:
            if col in df.columns:
                try:
                    row.append(float(df[col].iloc[0]))
                except Exception:
                    # if weird string or array-like, just set 0 for now
                    row.append(0.0)
            else:
                row.append(0.0)

        vectors.append(row)
        labels.append(site_label)

    return np.array(vectors, dtype=float), np.array(labels)

vn_X, vn_y = load_first_row_features(
    "../../data/PPG/*.csv",
    site_label="Vietnam"
)
mi_X, mi_y = load_first_row_features(
    "../../code_xg/preprocess_MIMIC-III/ppg_features_by_icu_v2/*.csv",
    site_label="MIMIC"
)

X = np.vstack([vn_X, mi_X])
y = np.concatenate([vn_y, mi_y])

print("Total samples:", X.shape)
print("Vietnam:", vn_X.shape, "MIMIC:", mi_X.shape)

# 1. Replace NaNs/infs if any
print("Before cleaning: NaNs =", np.isnan(X).sum(), "infs =", np.isinf(X).sum())
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print("After nan_to_num: NaNs =", np.isnan(X).sum(), "infs =", np.isinf(X).sum())

# 2. Clip extreme values to avoid overflow
CLIP = 1e6  # you can tighten to 1e4 if needed
X = np.clip(X, -CLIP, CLIP)
print("After clipping: min =", X.min(), "max =", X.max())

# 3. Drop zero-variance features
feat_var = X.var(axis=0)
keep = feat_var > 1e-12
print("Zero-variance features dropped:", np.sum(~keep))
X = X[:, keep]

if X.shape[1] == 0:
    raise ValueError("All features have zero variance after filtering.")

# 4. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before t-SNE: NaNs =", np.isnan(X_scaled).sum(), "infs =", np.isinf(X_scaled).sum())

# 5. t-SNE WITHOUT internal PCA
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',     # <--- important: no internal PCA
    random_state=42,
    verbose=1,
)

X_emb = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_emb[y == "Vietnam", 0],
    X_emb[y == "Vietnam", 1],
    c="blue", label="Vietnam", alpha=0.6
)
plt.scatter(
    X_emb[y == "MIMIC", 0],
    X_emb[y == "MIMIC", 1],
    c="red", label="MIMIC", alpha=0.6
)
plt.legend()
plt.title("t-SNE of PPG First-Row Feature Vectors (Vietnam vs MIMIC)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_ppg_vietnam_vs_mimic.png", dpi=300)
plt.show()
