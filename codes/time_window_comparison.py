import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from classification import (
    train_and_evaluate_classifier,
    perform_statistical_tests,
    compute_feature_importance,
    compute_shap_feature_importance,
)
from dataset_preprocess import DataPreprocessor
from feature_stat_comparison import MODALITY_BUILDERS, STAT_SETS
from utils import set_seed

warnings.filterwarnings("ignore")

TIME_CUTOFFS_HOURS = [2, 4, 6, 12, 24, None]  # None = full recording, no truncation
LABEL_NAME = 'DIED'

CLASSIFIERS = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

# Reduced modality scope (the combined feature sets only, not the singleton
# PPG/ECG/Vital_Signs/Clinical) -- reuses the same builders as
# feature_stat_comparison.py so feature lists stay identical between scripts.
MODALITIES = ['PPG+ECG', 'Clinical+PPG', 'Clinical+ECG', 'Clinical+PPG+ECG']


def run_full_grid(time_cutoff_hours, label_name=LABEL_NAME):
    """Run classifier x modality x stat-set at a single time cutoff, truncating
    PPG/ECG/vital-sign readings to the first `time_cutoff_hours` hours of each
    patient's recording before aggregation (None = use the full recording)."""
    set_seed(42)
    cutoff_label = 'full' if time_cutoff_hours is None else f'{time_cutoff_hours}h'
    print(f"\n{'=' * 80}\nTime-window cutoff: {cutoff_label}\n{'=' * 80}")

    use_pca = False
    n_components = 0.9

    feature_sets = {
        f'{modality}_{stat_name}': MODALITY_BUILDERS[modality](stats)
        for modality in MODALITIES
        for stat_name, stats in STAT_SETS.items()
    }

    split_method = 'kfold'
    kfold = 10
    num_splits = 200

    auroc_scores, auprc_scores = {}, {}
    summary_rows = []
    feature_list = None

    for modality in MODALITIES:
        for stat_name, stats in STAT_SETS.items():
            feature_set_name = f'{modality}_{stat_name}'
            feature_list = feature_sets[feature_set_name]
            print(f"\n--- Feature Set: {feature_set_name} ({len(feature_list)} features) ---")
            for clf_name, clf in CLASSIFIERS.items():
                combined_name = f"{feature_set_name}_{clf_name}"
                metrics_list, auroc_list, auprc_list, feature_names, _, _ = train_and_evaluate_classifier(
                    clf_name, copy.deepcopy(clf), label_name, 'none', split_method, kfold, num_splits,
                    use_pca, n_components, feature_list, use_RFE=False, RFE_feature_number=None,
                    time_cutoff_hours=time_cutoff_hours)
                sensitivity_list = [m['Sensitivity'] for m in metrics_list]
                specificity_list = [m['Specificity'] for m in metrics_list]
                f1_list = [m['F1-score'] for m in metrics_list]
                auroc_scores[combined_name] = auroc_list
                auprc_scores[combined_name] = auprc_list
                summary_rows.append({
                    'time_cutoff_hours': cutoff_label,
                    'feature_set': feature_set_name,
                    'modality': modality,
                    'stat_set': stat_name,
                    'classifier': clf_name,
                    'AUROC_mean': np.mean(auroc_list), 'AUROC_std': np.std(auroc_list),
                    'AUPRC_mean': np.mean(auprc_list), 'AUPRC_std': np.std(auprc_list),
                    'Sensitivity_mean': np.mean(sensitivity_list), 'Sensitivity_std': np.std(sensitivity_list),
                    'Specificity_mean': np.mean(specificity_list), 'Specificity_std': np.std(specificity_list),
                    'F1-score_mean': np.mean(f1_list), 'F1-score_std': np.std(f1_list),
                })

    classifiers_list = list(auroc_scores.keys())
    perform_statistical_tests(auroc_scores, classifiers_list, "AUROC")
    perform_statistical_tests(auprc_scores, classifiers_list, "AUPRC")

    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median', normalize=True,
                                    feature_list=feature_list, time_cutoff_hours=time_cutoff_hours)

    for clf_name, clf in CLASSIFIERS.items():
        compute_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

    for clf_name, clf in CLASSIFIERS.items():
        compute_shap_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

    return pd.DataFrame(summary_rows)


def plot_time_stat_lines(combined, metric_col, metric_label, out_path):
    """Grid of line plots: rows = classifier, columns = modality. Each panel
    shows `metric_col` vs. time cutoff, one line per stat set, so degradation
    patterns (e.g. slope_std needing more data points than mean_var) are
    visible directly."""
    classifiers_list = list(CLASSIFIERS.keys())
    stat_names = list(STAT_SETS.keys())
    cutoff_order = ['full' if c is None else f'{c}h' for c in TIME_CUTOFFS_HOURS]
    colors = dict(zip(stat_names, plt.cm.tab10.colors))
    std_col = metric_col.replace('_mean', '_std')

    fig, axes = plt.subplots(len(classifiers_list), len(MODALITIES),
                              figsize=(4.5 * len(MODALITIES), 4 * len(classifiers_list)),
                              squeeze=False, sharey=True)

    for row, clf_name in enumerate(classifiers_list):
        for col, modality in enumerate(MODALITIES):
            ax = axes[row][col]
            subset = combined[(combined['classifier'] == clf_name) & (combined['modality'] == modality)]
            for stat_name in stat_names:
                stat_data = (subset[subset['stat_set'] == stat_name]
                             .set_index('time_cutoff_hours').reindex(cutoff_order))
                ax.errorbar(cutoff_order, stat_data[metric_col], yerr=stat_data[std_col],
                            marker='o', markersize=4, capsize=3, label=stat_name, color=colors[stat_name])
            if row == 0:
                ax.set_title(modality)
            if col == 0:
                ax.set_ylabel(f'{clf_name}\n{metric_label}')
            ax.set_xlabel('Time-window cutoff')
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(stat_names), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f'{metric_label} vs. monitoring-time cutoff, by stat set / modality / classifier', y=1.10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    summaries = []
    for cutoff in TIME_CUTOFFS_HOURS:
        summaries.append(run_full_grid(cutoff))
        # Checkpoint after every cutoff so a crash partway through the sweep
        # (e.g. an out-of-memory error on a later cutoff) doesn't lose the
        # results already computed for earlier ones.
        pd.concat(summaries, ignore_index=True).to_csv('../data/time_window_comparison_results.csv', index=False)

    combined = pd.concat(summaries, ignore_index=True)

    cutoff_order = ['full' if c is None else f'{c}h' for c in TIME_CUTOFFS_HOURS]

    pivot_auroc = combined.pivot_table(index=['feature_set', 'classifier'],
                                        columns='time_cutoff_hours', values='AUROC_mean')[cutoff_order]
    pivot_auprc = combined.pivot_table(index=['feature_set', 'classifier'],
                                        columns='time_cutoff_hours', values='AUPRC_mean')[cutoff_order]
    pivot_sensitivity = combined.pivot_table(index=['feature_set', 'classifier'],
                                              columns='time_cutoff_hours', values='Sensitivity_mean')[cutoff_order]
    pivot_specificity = combined.pivot_table(index=['feature_set', 'classifier'],
                                              columns='time_cutoff_hours', values='Specificity_mean')[cutoff_order]
    pivot_f1 = combined.pivot_table(index=['feature_set', 'classifier'],
                                     columns='time_cutoff_hours', values='F1-score_mean')[cutoff_order]

    print(f"\n\n{'=' * 80}\nAUROC comparison across time-window cutoffs\n{'=' * 80}")
    print(pivot_auroc.round(3).to_string())
    print(f"\n{'=' * 80}\nAUPRC comparison across time-window cutoffs\n{'=' * 80}")
    print(pivot_auprc.round(3).to_string())
    print(f"\n{'=' * 80}\nSensitivity comparison across time-window cutoffs\n{'=' * 80}")
    print(pivot_sensitivity.round(3).to_string())
    print(f"\n{'=' * 80}\nSpecificity comparison across time-window cutoffs\n{'=' * 80}")
    print(pivot_specificity.round(3).to_string())
    print(f"\n{'=' * 80}\nF1-score comparison across time-window cutoffs\n{'=' * 80}")
    print(pivot_f1.round(3).to_string())

    plot_time_stat_lines(combined, 'AUROC_mean', 'AUROC', '../data/time_window_comparison_auroc.png')
    plot_time_stat_lines(combined, 'AUPRC_mean', 'AUPRC', '../data/time_window_comparison_auprc.png')
    plot_time_stat_lines(combined, 'Specificity_mean', 'Specificity', '../data/time_window_comparison_specificity.png')
    plot_time_stat_lines(combined, 'F1-score_mean', 'F1-score', '../data/time_window_comparison_f1.png')

    print("\nFull results written to ../data/time_window_comparison_results.csv")
    print("AUROC plot written to ../data/time_window_comparison_auroc.png")
    print("AUPRC plot written to ../data/time_window_comparison_auprc.png")
    print("Specificity plot written to ../data/time_window_comparison_specificity.png")
    print("F1-score plot written to ../data/time_window_comparison_f1.png")


if __name__ == "__main__":
    main()
