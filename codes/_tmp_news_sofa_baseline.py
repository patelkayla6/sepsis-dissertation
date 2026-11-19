"""Temporary runner: AUROC/AUPRC/Sensitivity/Specificity for raw NEWS/SOFA baselines only
(no classifier training). Mirrors the baseline loop in classification.py's main(),
using the same split params (kfold=10, num_splits=200) so results are directly
comparable to a full classification.py run."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from classification import evaluate_raw_score_baseline
from utils import set_seed, sofa_feature_names

set_seed(42)

label_name = 'DIED'
split_method = 'kfold'
kfold = 10
num_splits = 200

rows = []
for score_col in sofa_feature_names:
    auroc_list, auprc_list, sensitivity_list, specificity_list, f1_list, _, _ = evaluate_raw_score_baseline(
        score_col, label_name, split_method, kfold, num_splits)
    rows.append({
        'score': score_col,
        'AUROC_mean': np.mean(auroc_list), 'AUROC_std': np.std(auroc_list),
        'AUPRC_mean': np.mean(auprc_list), 'AUPRC_std': np.std(auprc_list),
        'Sensitivity_mean': np.mean(sensitivity_list), 'Sensitivity_std': np.std(sensitivity_list),
        'Specificity_mean': np.mean(specificity_list), 'Specificity_std': np.std(specificity_list),
    })

df = pd.DataFrame(rows)
df.to_csv('../data/news_sofa_baseline_results.csv', index=False)
print(df.round(3).to_string(index=False))
print("\nWritten to ../data/news_sofa_baseline_results.csv")
