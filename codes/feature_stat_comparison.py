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
from utils import (
    set_seed, stat_feature_names,
    normal_feature_names, static_vital_sign_feature_names,
    ppg_base_names, ecg_base_names, vital_base_names,
)

warnings.filterwarnings("ignore")

LABEL_NAME = 'DIED'

# 'Clinical' matches classification.py's definition exactly (normal_feature_names,
# which bundles in the continuously monitored MONITORING_*_mean/var columns) so
# results are directly comparable across the two scripts. This means it overlaps
# with the standalone 'Vital_Signs' modality below -- Clinical+PPG/ECG combos
# always carry mean/var vitals regardless of which stat_set is being tested for
# PPG/ECG, weakening the stat-pair ablation for anything that includes Clinical.
STAT_SETS = {
    'mean_var':   ['mean', 'var'],
    'median_mad': ['median', 'mad'],
    'slope_std':  ['slope', 'std'],
}

# Each modality is a function of the stat pair, so callers can build its
# feature list for whichever of STAT_SETS is being tested. Modalities built
# purely from static fields (Clinical) ignore the `stats` argument -- their
# feature list is identical across every stat pair.
MODALITY_BUILDERS = {
    'Vital_Signs':       lambda stats: static_vital_sign_feature_names + stat_feature_names(vital_base_names, stats),
    'Clinical':          lambda stats: normal_feature_names,
    'PPG':               lambda stats: stat_feature_names(ppg_base_names, stats),
    'ECG':               lambda stats: stat_feature_names(ecg_base_names, stats),
    'PPG+ECG':           lambda stats: stat_feature_names(ppg_base_names, stats) + stat_feature_names(ecg_base_names, stats),
    'Clinical+PPG':      lambda stats: normal_feature_names + stat_feature_names(ppg_base_names, stats),
    'Clinical+ECG':      lambda stats: normal_feature_names + stat_feature_names(ecg_base_names, stats),
    'Clinical+PPG+ECG':  lambda stats: normal_feature_names + stat_feature_names(ppg_base_names, stats) + stat_feature_names(ecg_base_names, stats),
}

# Modalities with no aggregation stat to vary (built purely from static
# fields) are only run once instead of once per entry in STAT_SETS.
STAT_INVARIANT_MODALITIES = {'Clinical'}


def build_feature_sets():
    """Returns (feature_sets, feature_set_meta), where feature_set_meta maps each
    feature_set name to its (modality, stat_set) origin -- stat_set is None for
    modalities in STAT_INVARIANT_MODALITIES, which don't vary by aggregation stat."""
    feature_sets = {}
    feature_set_meta = {}
    for modality, build_fn in MODALITY_BUILDERS.items():
        if modality in STAT_INVARIANT_MODALITIES:
            feature_sets[modality] = build_fn(stats=None)
            feature_set_meta[modality] = (modality, None)
        else:
            for stat_name, stats in STAT_SETS.items():
                name = f'{modality}_{stat_name}'
                feature_sets[name] = build_fn(stats)
                feature_set_meta[name] = (modality, stat_name)
    return feature_sets, feature_set_meta


def plot_metric_heatmaps(combined, metric_col, metric_label, out_path):
    """One heatmap per classifier for the given metric: modality (rows) x
    aggregation-stat set (columns). Modalities with no stat variant (e.g.
    Clinical) are repeated across every stat column so they appear as a
    constant reference row in each panel."""
    modalities = list(MODALITY_BUILDERS.keys())
    stat_names = list(STAT_SETS.keys())
    classifiers_list = combined['classifier'].unique().tolist()

    expanded_rows = []
    for _, row in combined.iterrows():
        if pd.isna(row['stat_set']):
            for stat_name in stat_names:
                expanded_rows.append({**row.to_dict(), 'stat_set': stat_name})
        else:
            expanded_rows.append(row.to_dict())
    expanded = pd.DataFrame(expanded_rows)

    ncols = 2
    nrows = -(-len(classifiers_list) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows), squeeze=False)
    vmin, vmax = expanded[metric_col].min(), expanded[metric_col].max()
    im = None

    for idx, clf_name in enumerate(classifiers_list):
        ax = axes[idx // ncols][idx % ncols]
        pivot = expanded[expanded['classifier'] == clf_name].pivot_table(
            index='modality', columns='stat_set', values=metric_col
        ).reindex(index=modalities, columns=stat_names)

        im = ax.imshow(pivot.values, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(stat_names)))
        ax.set_xticklabels(stat_names, rotation=30, ha='right')
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels(modalities)
        ax.set_title(clf_name)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                            color=text_color, fontsize=8)

    for idx in range(len(classifiers_list), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.subplots_adjust(hspace=0.6, wspace=0.7, top=0.90, right=0.85)
    fig.colorbar(im, ax=axes, shrink=0.8, label=metric_label)
    fig.suptitle(f'{metric_label} by modality x aggregation-stat set, per classifier')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(label_name=LABEL_NAME):
    set_seed(42)

    classifiers = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    use_pca = False
    n_components = 0.9

    feature_sets, feature_set_meta = build_feature_sets()

    split_method = 'kfold'
    kfold = 10
    num_splits = 200

    auroc_scores, auprc_scores = {}, {}
    summary_rows = []
    feature_list = None

    for feature_set_name, feature_list in feature_sets.items():
        modality, stat_set = feature_set_meta[feature_set_name]
        print(f"\n--- Feature Set: {feature_set_name} ({len(feature_list)} features) ---")
        for clf_name, clf in classifiers.items():
            combined_name = f"{feature_set_name}_{clf_name}"
            metrics_list, auroc_list, auprc_list, feature_names, _, _ = train_and_evaluate_classifier(
                clf_name, copy.deepcopy(clf), label_name, 'none', split_method, kfold, num_splits,
                use_pca, n_components, feature_list, use_RFE=False, RFE_feature_number=None)
            sensitivity_list = [m['Sensitivity'] for m in metrics_list]
            specificity_list = [m['Specificity'] for m in metrics_list]
            f1_list = [m['F1-score'] for m in metrics_list]
            auroc_scores[combined_name] = auroc_list
            auprc_scores[combined_name] = auprc_list
            summary_rows.append({
                'feature_set': feature_set_name,
                'modality': modality,
                'stat_set': stat_set,
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
                                    feature_list=feature_list)

    for clf_name, clf in classifiers.items():
        if clf_name in ['Logistic_Regression', 'Random_Forest', 'XGBoost']:
            compute_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

    for clf_name, clf in classifiers.items():
        compute_shap_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

    combined = pd.DataFrame(summary_rows)
    combined.to_pickle('../data/feature_stat_comparison_results.pkl')  # safety checkpoint in case to_csv/heatmap saving fails
    combined.to_csv('../data/feature_stat_comparison_results.csv', index=False)

    pivot_auroc = combined.pivot_table(index='classifier', columns='feature_set', values='AUROC_mean')
    pivot_auprc = combined.pivot_table(index='classifier', columns='feature_set', values='AUPRC_mean')
    pivot_sensitivity = combined.pivot_table(index='classifier', columns='feature_set', values='Sensitivity_mean')
    pivot_specificity = combined.pivot_table(index='classifier', columns='feature_set', values='Specificity_mean')
    pivot_f1 = combined.pivot_table(index='classifier', columns='feature_set', values='F1-score_mean')

    print(f"\n\n{'=' * 80}\nAUROC comparison across feature sets\n{'=' * 80}")
    print(pivot_auroc.round(3).to_string())
    print(f"\n{'=' * 80}\nAUPRC comparison across feature sets\n{'=' * 80}")
    print(pivot_auprc.round(3).to_string())
    print(f"\n{'=' * 80}\nSensitivity comparison across feature sets\n{'=' * 80}")
    print(pivot_sensitivity.round(3).to_string())
    print(f"\n{'=' * 80}\nSpecificity comparison across feature sets\n{'=' * 80}")
    print(pivot_specificity.round(3).to_string())
    print(f"\n{'=' * 80}\nF1-score comparison across feature sets\n{'=' * 80}")
    print(pivot_f1.round(3).to_string())

    plot_metric_heatmaps(combined, 'AUROC_mean', 'AUROC', '../data/feature_stat_comparison_auroc_heatmap.png')
    plot_metric_heatmaps(combined, 'AUPRC_mean', 'AUPRC', '../data/feature_stat_comparison_auprc_heatmap.png')
    plot_metric_heatmaps(combined, 'Sensitivity_mean', 'Sensitivity',
                          '../data/feature_stat_comparison_sensitivity_heatmap.png')
    plot_metric_heatmaps(combined, 'Specificity_mean', 'Specificity',
                          '../data/feature_stat_comparison_specificity_heatmap.png')
    plot_metric_heatmaps(combined, 'F1-score_mean', 'F1-score',
                          '../data/feature_stat_comparison_f1_heatmap.png')

    print("\nFull results written to ../data/feature_stat_comparison_results.csv")
    print("AUROC heatmap written to ../data/feature_stat_comparison_auroc_heatmap.png")
    print("AUPRC heatmap written to ../data/feature_stat_comparison_auprc_heatmap.png")
    print("Sensitivity heatmap written to ../data/feature_stat_comparison_sensitivity_heatmap.png")
    print("Specificity heatmap written to ../data/feature_stat_comparison_specificity_heatmap.png")
    print("F1-score heatmap written to ../data/feature_stat_comparison_f1_heatmap.png")


if __name__ == "__main__":
    main()
