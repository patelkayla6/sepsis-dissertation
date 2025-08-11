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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Feature Bases ---
PPG_feature_bases = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
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
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC'
    'Tpi', 'Tpp', 'Tsys', 'Tdia', 'Tsp', 'Tdp', 'deltaT',
     'Tsw10', 'Tsw25', 'Tsw33', 'Tsw50', 'Tsw66', 'Tsw75', 'Tsw90',
     'Tdw10', 'Tdw25', 'Tdw33', 'Tdw50', 'Tdw66', 'Tdw75', 'Tdw90',
     'Tpw10', 'Tpw25', 'Tpw33', 'Tpw50', 'Tpw66', 'Tpw75', 'Tpw90',
     'Asp', 'Adn', 'Adp', 'Aoff', 'AUCpi', 'AUCsys', 'AUCdia', ]

# --- Data Loading & Preprocessing ---
def load_patient_signal(file_path, feats):
    """
    Load raw columns, no normalization or imputation.
    Return a float32 numpy array with NaNs preserved (∞ → NaN).
    This keeps behavior aligned with dataset_preprocess.load_signal_features,
    where stats are computed after dropping NaNs/∞ per feature.
    """
    df = pd.read_csv(file_path, usecols=lambda c: (c in feats))
    # ensure numeric & unify bad values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
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
    """
    For each file, compute per-feature:
      mean, var (ddof=0), skew, kurtosis, median
    after dropping NaNs/∞ for that feature only.
    Concatenate stats in the order:
      [f1_mean, f1_var, f1_skew, f1_kurt, f1_median, f2_mean, ...]
    """
    stats, ids = [], []
    for path in tqdm(csv_paths, desc="Extracting stats (mean/var/skew/kurt/median)"):
        X = load_patient_signal(path, feats)   # shape [T, D] with NaNs
        if X.size == 0:
            continue

        feat_vec = []
        D = X.shape[1]
        for j in range(D):
            col = X[:, j]
            # drop NaN / inf (already mapped to NaN)
            valid = np.isfinite(col) & ~np.isnan(col)
            y = col[valid]

            if y.size == 0:
                m = v = s = k = med = np.nan
            else:
                m   = float(np.mean(y))
                v   = float(np.var(y))            # ddof=0 to match np.var default in your script
                # s   = float(skew(y, bias=True))   # skew/kurt like your code (nan_policy not needed after filtering)
                # k   = float(kurtosis(y, bias=True))
                # med = float(np.median(y))

            feat_vec.extend([m, v])

        stats.append(np.array(feat_vec, dtype=np.float32))
        ids.append(os.path.basename(path).replace(".csv", ""))

    return np.stack(stats), ids

# --- Main ---
if __name__ == '__main__':
    # --- Load model ---
    model = GPTFeatureModel(input_dim=len(PPG_feature_bases), max_seq_len=180).to(device)
    # model.load_state_dict(torch.load("gpt_feature_model.pt"))
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

    # --- Prepare final data ---
    X = dataset.filter(like="feat_").values
    y = dataset[label_name].values

    n_repeats = 20
    n_splits = 10
    random_state = 42

    aucs, auprcs, accs, sens_list, spec_list, f1s = [], [], [], [], [], []

    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(random_state=42))
            ])
            clf.fit(X_train, y_train)

            # If you want to replicate the "optimal threshold" behavior from your original code:
            # compute threshold on training data and apply to test predictions.
            y_prob_train = clf.predict_proba(X_train)[:, 1]
            fpr, tpr, thr = roc_curve(y_train, y_prob_train)
            opt_thr = thr[(tpr - fpr).argmax()]

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= opt_thr).astype(int)

            # aggregate
            y_true_all.extend(y_test.tolist())
            y_prob_all.extend(y_prob.tolist())
            y_pred_all.extend(y_pred.tolist())

        # compute metrics ONCE per repeat using all test predictions in that repeat
        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        y_pred_all = np.array(y_pred_all)

        # Metrics (aggregated per repeat)
        auc = roc_auc_score(y_true_all, y_prob_all)
        auprc = average_precision_score(y_true_all, y_prob_all)
        acc = (y_pred_all == y_true_all).mean()
        sens = recall_score(y_true_all, y_pred_all, zero_division=0)  # sensitivity = TPR
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity = TNR
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

        aucs.append(auc);
        auprcs.append(auprc)
        accs.append(acc);
        sens_list.append(sens);
        spec_list.append(spec);
        f1s.append(f1)

        print(f"Repeat {repeat + 1}/{n_repeats} — AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | "
              f"Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")

    # Summary
    print(f"\n=== Repeated {n_splits}-Fold CV (mean ± std over {n_repeats} repeats) ===")
    print(f"AUROC    : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AUPRC    : {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Sensitivity (TPR): {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}")
    print(f"Specificity (TNR): {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
    print(f"F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_2d = tsne.fit_transform(X)
    #
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='coolwarm', s=40)
    # plt.title("t-SNE projection of feature embeddings")
    # plt.xlabel("Dim 1");
    # plt.ylabel("Dim 2")
    # plt.legend(title="Label")
    # plt.tight_layout()
    # plt.show()
