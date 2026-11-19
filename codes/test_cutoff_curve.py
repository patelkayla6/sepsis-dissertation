import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from classification import train_and_evaluate_classifier
from utils import set_seed, normal_feature_names, PPG_feature_names, ECG_feature_names

warnings.filterwarnings("ignore")

# Model is always trained on full recordings; only the test-time window varies,
# swept densely to trace out a continuous performance-vs-monitoring-time curve
# (rather than the handful of discrete cutoffs in train_full_test_truncated_comparison.py).
TRAIN_TIME_CUTOFF_HOURS = None
TEST_TIME_CUTOFFS_HOURS = [round(h, 1) for h in np.arange(0.5, 24.5, 0.5)]  # 0.5h .. 24.0h, every 30 min
LABEL_NAME = 'DIED'

# Reduced grid: only the classifiers/feature sets asked for, since a smooth curve
# doesn't need the full 7x4 grid and this sweep already multiplies runtime by the
# number of cutoffs.
CLASSIFIERS = {
    'Random_Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

FEATURE_SETS = {
    'PPG+ECG':          PPG_feature_names + ECG_feature_names,
    'Clinical+PPG':     normal_feature_names + PPG_feature_names,
    'Clinical+ECG':     normal_feature_names + ECG_feature_names,
    'Clinical+PPG+ECG': normal_feature_names + PPG_feature_names + ECG_feature_names,
}

SPLIT_METHOD = 'kfold'
KFOLD = 10
NUM_SPLITS = 200


def run_point(test_time_cutoff_hours):
    """Run the reduced grid (2 classifiers x 4 feature sets) at a single test-time
    cutoff and return one summary row per combo. No statistical tests, feature
    importance, or SHAP here -- those describe the trained model itself or compare
    combos to each other, neither of which changes across test cutoffs, so running
    them at every point in a dense sweep would just repeat the same work."""
    set_seed(42)
    rows = []
    for feature_set_name, feature_list in FEATURE_SETS.items():
        for clf_name, clf in CLASSIFIERS.items():
            _, auroc_list, auprc_list, _, _, _ = train_and_evaluate_classifier(
                clf_name, copy.deepcopy(clf), LABEL_NAME, 'none', SPLIT_METHOD, KFOLD, NUM_SPLITS,
                use_pca=False, n_components=None, feature_list=feature_list,
                use_RFE=False, RFE_feature_number=None,
                time_cutoff_hours=TRAIN_TIME_CUTOFF_HOURS, test_time_cutoff_hours=test_time_cutoff_hours)
            rows.append({
                'test_time_cutoff_hours': test_time_cutoff_hours,
                'feature_set': feature_set_name,
                'classifier': clf_name,
                'AUROC_mean': np.mean(auroc_list), 'AUROC_std': np.std(auroc_list),
                'AUPRC_mean': np.mean(auprc_list), 'AUPRC_std': np.std(auprc_list),
            })
    return rows


def plot_curve(combined, metric, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (feature_set_name, clf_name), group in combined.groupby(['feature_set', 'classifier']):
        group = group.sort_values('test_time_cutoff_hours')
        label = f"{feature_set_name} / {clf_name}"
        ax.plot(group['test_time_cutoff_hours'], group[f'{metric}_mean'], marker='o', markersize=2, label=label)
    ax.set_xlabel('Test-time window (hours of monitoring available)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs. test-time monitoring window (trained on full recordings)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print(f"Sweeping {len(TEST_TIME_CUTOFFS_HOURS)} test-time cutoffs "
          f"({TEST_TIME_CUTOFFS_HOURS[0]}h to {TEST_TIME_CUTOFFS_HOURS[-1]}h) "
          f"over {len(FEATURE_SETS)} feature sets x {len(CLASSIFIERS)} classifiers...")

    all_rows = []
    for i, cutoff in enumerate(TEST_TIME_CUTOFFS_HOURS, 1):
        print(f"\n[{i}/{len(TEST_TIME_CUTOFFS_HOURS)}] test cutoff = {cutoff}h")
        all_rows.extend(run_point(cutoff))

    combined = pd.DataFrame(all_rows)
    combined.to_csv('../data/test_cutoff_curve_results.csv', index=False)

    plot_curve(combined, 'AUROC', '../data/test_cutoff_curve_auroc.png')
    plot_curve(combined, 'AUPRC', '../data/test_cutoff_curve_auprc.png')

    print("\nResults written to ../data/test_cutoff_curve_results.csv")
    print("Plots written to ../data/test_cutoff_curve_auroc.png and ../data/test_cutoff_curve_auprc.png")


if __name__ == "__main__":
    main()
