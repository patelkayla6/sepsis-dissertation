import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from x_transformers import Decoder

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

# --- Data Loading & Preprocessing ---
def load_patient_signal(file_path, feats):
    df = pd.read_csv(file_path, usecols=feats)
    # Step 1: ffill then fill remaining NaNs with column mean
    df = df.fillna(method='ffill').fillna(df.mean())

    # Step 2: Z-score normalization for each column
    for col in df.columns:
        m = df[col].mean()
        s = df[col].std()
        df[col] = (df[col] - m) / (s + 1e-6)
        df[col] = df[col].fillna(0.0)  # Handle potential post-norm NaNs

    return df.to_numpy(dtype=np.float32)

from model import GPTFeatureModel

# --- Model ---
class GPTFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model: GPTFeatureModel):
        super().__init__()
        self.input_proj = pretrained_model.input_proj
        self.pos_encoding = pretrained_model.pos_encoding
        self.gpt = pretrained_model.gpt
        self.norm = pretrained_model.norm
        self.to_latent = pretrained_model.to_latent

    def forward(self, x):
        """
        Args:
            x: [B, T, D] — full sequence
        Returns:
            z: [B, T, proj_dim] — latent representation at each step
        """
        B, T, D = x.shape
        h = self.input_proj(x.view(-1, D)).view(B, T, -1)
        h = self.pos_encoding(h)
        h = self.gpt(h)
        h = self.norm(h)
        z = self.to_latent(h)
        return z

    def extract_last_step_feature(self, x):
        """
        Returns:
            z_last: [B, proj_dim] — feature at the last time step
        """
        z = self.forward(x)
        return z[:, -1, :]


# --- Embedding Extraction ---
def extract_gpt_embeddings(model, csv_paths, feats, window_len=180):
    model.eval()
    embeddings, ids = [], []
    for path in tqdm(csv_paths, desc="Extracting GPT embeddings"):
        x = load_patient_signal(path, feats)
        if len(x) < window_len: continue
        x_tensor = torch.tensor(x[:window_len]).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model.extract_last_step_feature(x_tensor)
        embeddings.append(z.squeeze(0).cpu().numpy())
        ids.append(os.path.basename(path).replace(".csv", ""))
    return np.stack(embeddings), ids

def extract_stat_features(model, csv_paths, feats, window_len=180):
    stats, ids = [], []
    for path in tqdm(csv_paths, desc="Extracting mean+std features"):
        x = load_patient_signal(path, feats)
        # if len(x) < window_len: continue
        # x_window = x[:window_len]
        x_window = x
        mean_vec = np.mean(x_window, axis=0)
        std_vec = np.std(x_window, axis=0)

        stats.append(np.concatenate([mean_vec, std_vec]))
        ids.append(os.path.basename(path).replace(".csv", ""))
    return np.stack(stats), ids

# --- Main ---
if __name__ == '__main__':
    # --- Load model ---
    model = GPTFeatureModel(input_dim=len(PPG_feature_bases), max_seq_len=180).to(device)
    model.load_state_dict(torch.load("gpt_feature_model.pt"))
    model = GPTFeatureExtractor(model).to(device)
    model.eval()

    # --- Input data ---
    csv_paths = sorted(glob.glob("../data/PPG/*.csv"))
    patient_ids = [os.path.basename(p).replace(".csv", "") for p in csv_paths]

    # --- Extract embeddings ---
    X, ids = extract_stat_features(model, csv_paths, PPG_feature_bases) # extract_gpt_embeddings(model, csv_paths, PPG_feature_bases)
    features_df = pd.DataFrame(X, index=ids, columns=[f"feat_{i}" for i in range(X.shape[1])])
    features_df.index = features_df.index.str.replace(r'^.*?-', '', regex=True)  # remove prefix like '01NVa-'
    features_df.index = features_df.index.str.replace(r'_.*$', '', regex=True)  # remove suffix like '_20s'
    features_df.index.name = 'USUBJID'

    # --- Load labels ---
    print("Loading labels...")
    excel_data = pd.ExcelFile('../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx')
    labels = pd.read_excel(excel_data, 'CLI')[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
    labels = labels.astype({col: int for col in ['DIED', 'HFNOV', 'MEVEN']})
    labels = labels.set_index('USUBJID')

    # --- Join features and labels ---
    dataset = features_df.join(labels, how='inner')

    # --- Remove specific IDs (for DIED label only) ---
    label_name = 'DIED'
    if label_name == 'DIED':
        remove_ids = ['003-1011', '003-1016', '003-1017', '003-1214', '003-1219', '003-1239']
        dataset = dataset[~dataset.index.isin(remove_ids)]

    dataset.fillna(0., inplace=True)
    # --- Prepare final data ---
    X = dataset.filter(like="feat_").values

    y = dataset[label_name].values

    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    n_repeats = 10
    n_splits = 10
    random_state = 42

    aucs, accs, precisions, recalls, f1s = [], [], [], [], []

    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat + random_state)

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        acc_fold, prec_fold, rec_fold, f1_fold = [], [], [], []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)

            acc_fold.append(accuracy_score(y_test, y_pred))
            prec_fold.append(precision_score(y_test, y_pred, zero_division=0))
            rec_fold.append(recall_score(y_test, y_pred, zero_division=0))
            f1_fold.append(f1_score(y_test, y_pred, zero_division=0))

        # Compute AUROC once per repeat (full set)
        auc = roc_auc_score(y_true_all, y_prob_all)
        aucs.append(auc)
        accs.append(np.mean(acc_fold))
        precisions.append(np.mean(prec_fold))
        recalls.append(np.mean(rec_fold))
        f1s.append(np.mean(f1_fold))

        print(f"Repeat {repeat + 1}/{n_repeats} AUROC: {auc:.4f}")

    # Summary
    print(f"\n=== Repeated {n_splits}-Fold CV (mean over {n_repeats} repeats) ===")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='coolwarm', s=40)
    plt.title("t-SNE projection of feature embeddings")
    plt.xlabel("Dim 1");
    plt.ylabel("Dim 2")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()
