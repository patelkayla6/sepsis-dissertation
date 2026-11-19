import copy
import os
import pickle
import shutil
import time
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

from classification import process_single_split
from dataset_preprocess import DataPreprocessor
from utils import set_seed, normal_feature_names, PPG_feature_names, ECG_feature_names

warnings.filterwarnings("ignore")

# Model is always trained on full recordings; only the test-time window varies,
# generating a per-patient risk-score trajectory across cutoffs (same sweep as
# test_cutoff_curve.py, but here we keep each score tagged by patient ID instead
# of collapsing straight to aggregate AUROC/AUPRC).
TRAIN_TIME_CUTOFF_HOURS = None
# Hourly rather than half-hourly: clinical vitals (DAILY_VITAL) only update
# roughly once per hour (78.6% of consecutive readings are exactly 1.0h apart),
# so a 0.5h grid mostly duplicated clinical values between adjacent points for
# any Clinical-containing feature set, without adding real new information.
TEST_TIME_CUTOFFS_HOURS = [float(h) for h in range(1, 25)]  # 1h .. 24h, every hour
LABEL_NAME = 'DIED'

# Local pairing window for the stability metric (CAREBench-style local-Lipschitz
# constant): only cutoff pairs at most this many hours apart count as "local"
# comparisons. The paper text says c=10min, but CAREBench's own released code
# (evaluation/lipschitz.py) actually uses a 1.0-hour window on its hourly-spaced
# grids -- matched here too, which at our now-hourly grid means each patient's
# Lc is averaged over strictly-adjacent cutoff pairs only.
LOCAL_WINDOW_HOURS = 1.0

CLASSIFIERS = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

FEATURE_SETS = {
    'PPG':              PPG_feature_names,
    'ECG':              ECG_feature_names,
    'PPG+ECG':          PPG_feature_names + ECG_feature_names,
    'Clinical+PPG':     normal_feature_names + PPG_feature_names,
    'Clinical+ECG':     normal_feature_names + ECG_feature_names,
    'Clinical+PPG+ECG': normal_feature_names + PPG_feature_names + ECG_feature_names,
}

SPLIT_METHOD = 'kfold'
KFOLD = 10
NUM_SPLITS = 200

# Checkpointing: a full sweep is tens of thousands of model fits and can run for
# hours, so progress is pickled as it goes -- a crash or interrupt loses at most
# the current cutoff's work, not the whole run. Cleared automatically once main()
# finishes successfully (see main()), so stale checkpoints don't silently get
# reused after the metric logic itself changes.
CHECKPOINT_DIR = '../data/stability_checkpoints'


def _safe_name(feature_set_name):
    return feature_set_name.replace('+', '_')


def _progress_checkpoint_path(feature_set_name):
    """Mid-sweep checkpoint: partial scores/preds for a feature set still being
    computed, keyed by which cutoffs are done so far."""
    return os.path.join(CHECKPOINT_DIR, f'{_safe_name(feature_set_name)}_progress.pkl')


def _done_checkpoint_path(feature_set_name):
    """Whole-feature-set checkpoint: written once run_feature_set fully finishes,
    so a crash on a *later* feature set doesn't force redoing this one."""
    return os.path.join(CHECKPOINT_DIR, f'{_safe_name(feature_set_name)}_done.pkl')


def collect_trajectories(feature_list, feature_set_name):
    """Run the same kfold splits used for accuracy evaluation at every test-time
    cutoff, keeping each prediction tagged by patient ID instead of concatenating
    them into an anonymous array (which is what train_and_evaluate_classifier does).

    Returns {classifier_name: {cv_repeat: (score_df, pred_df)}} where score_df /
    pred_df have one row per patient (index = patient ID) and one column per test
    cutoff hour. score_df holds the continuous risk score (used for Lc). pred_df
    holds the binary alert from that fold's own decision threshold -- i.e.
    classification.py's compute_optimal_threshold, which is now recall-constrained
    (highest threshold that still hits >=90% recall on the training fold) rather
    than a fixed clinical constant -- used for flip count, so flip count reflects
    the same decision rule the rest of the pipeline uses for Sensitivity/F1.
    """
    num_CVs = NUM_SPLITS // KFOLD
    progress_path = _progress_checkpoint_path(feature_set_name)

    scores = {clf_name: {cv_num: {} for cv_num in range(num_CVs)} for clf_name in CLASSIFIERS}
    preds = {clf_name: {cv_num: {} for cv_num in range(num_CVs)} for clf_name in CLASSIFIERS}
    completed_cutoffs = set()

    if os.path.exists(progress_path):
        with open(progress_path, 'rb') as f:
            checkpoint = pickle.load(f)
        scores, preds, completed_cutoffs = checkpoint['scores'], checkpoint['preds'], checkpoint['completed_cutoffs']
        print(f"  Resuming from checkpoint: {len(completed_cutoffs)}/{len(TEST_TIME_CUTOFFS_HOURS)} cutoffs already done")

    total_cutoffs = len(TEST_TIME_CUTOFFS_HOURS)
    newly_done = 0
    start_time = time.time()
    for i, cutoff in enumerate(TEST_TIME_CUTOFFS_HOURS, 1):
        if cutoff in completed_cutoffs:
            print(f"  [{i}/{total_cutoffs}] cutoff={cutoff:.0f}h already done (checkpoint), skipping")
            continue

        # Shared across classifiers/cv-repeats for this cutoff -- load_excel is
        # lru_cached, so this mainly avoids re-running the imputer/scaler setup.
        preprocessor = DataPreprocessor(label_name=LABEL_NAME, missing_method='median', normalize=True,
                                        feature_list=feature_list, time_cutoff_hours=TRAIN_TIME_CUTOFF_HOURS,
                                        test_time_cutoff_hours=cutoff)
        for clf_name, clf in CLASSIFIERS.items():
            for cv_num in range(num_CVs):
                fold_scores, fold_preds = [], []
                for fold_num in range(KFOLD):
                    split_index = cv_num * KFOLD + fold_num
                    preprocessor.split_data(method=SPLIT_METHOD, n_splits=KFOLD, split_index=split_index)
                    y_test, y_prob_cal, y_pred, _ = process_single_split(
                        preprocessor, copy.deepcopy(clf), 'none', use_pca=False, n_components=None,
                        feature_names=None, use_RFE=False, RFE_feature_number=None)
                    fold_scores.append(pd.Series(y_prob_cal, index=y_test.index))
                    fold_preds.append(pd.Series(y_pred, index=y_test.index))
                scores[clf_name][cv_num][cutoff] = pd.concat(fold_scores)
                preds[clf_name][cv_num][cutoff] = pd.concat(fold_preds)

        completed_cutoffs.add(cutoff)
        newly_done += 1
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(progress_path, 'wb') as f:
            pickle.dump({'scores': scores, 'preds': preds, 'completed_cutoffs': completed_cutoffs}, f)

        elapsed = time.time() - start_time
        eta = elapsed / newly_done * (total_cutoffs - len(completed_cutoffs))
        print(f"  [{i}/{total_cutoffs}] cutoff={cutoff:.0f}h done "
              f"({elapsed:.0f}s elapsed since resume, ~{eta:.0f}s left for this feature set)")

    result = {}
    for clf_name in CLASSIFIERS:
        result[clf_name] = {}
        for cv_num in range(num_CVs):
            score_df = pd.DataFrame(scores[clf_name][cv_num]).sort_index(axis=1)
            pred_df = pd.DataFrame(preds[clf_name][cv_num]).sort_index(axis=1)
            result[clf_name][cv_num] = (score_df, pred_df)

    # All cutoffs done -- the run_feature_set-level "done" checkpoint takes over
    # as the durable record for this feature set, so the mid-sweep one is stale now.
    if os.path.exists(progress_path):
        os.remove(progress_path)

    return result


def compute_lc(traj_row, local_window_hours=LOCAL_WINDOW_HOURS):
    """CAREBench-style local-Lipschitz stability score for one patient's risk
    trajectory: mean |f(t) - f(t')| / |t - t'| over cutoff pairs within
    `local_window_hours` of each other. Lower = more stable."""
    valid = traj_row.dropna()
    times = valid.index.to_numpy(dtype=float)
    values = valid.to_numpy(dtype=float)
    diffs = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            dt = times[j] - times[i]
            if dt > local_window_hours:
                break  # times is sorted, so no further j can be within window
            diffs.append(abs(values[j] - values[i]) / dt)
    return float(np.mean(diffs)) if diffs else np.nan


def compute_flip_count(alert_row):
    """Number of 0<->1 transitions across one patient's binary alert trajectory,
    in order of increasing test cutoff. The alert is already thresholded per-fold
    by process_single_split using classification.py's recall-constrained
    compute_optimal_threshold, not a fixed cutoff -- so this reflects the same
    decision rule the rest of the pipeline uses for Sensitivity/F1."""
    valid = alert_row.dropna().sort_index()
    if len(valid) < 2:
        return np.nan
    return float(np.abs(np.diff(valid.to_numpy())).sum())


def run_feature_set(feature_set_name, feature_list):
    done_path = _done_checkpoint_path(feature_set_name)
    if os.path.exists(done_path):
        print(f"\n--- Feature Set: {feature_set_name} --- (loaded from checkpoint, already complete)")
        with open(done_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n--- Feature Set: {feature_set_name} ---")
    set_seed(42)
    trajectories = collect_trajectories(feature_list, feature_set_name)

    summary_rows, patient_rows = [], []
    for clf_name, cv_dict in trajectories.items():
        for cv_num, (score_df, pred_df) in cv_dict.items():
            patient_lc = score_df.apply(compute_lc, axis=1)
            patient_flip = pred_df.apply(compute_flip_count, axis=1)

            summary_rows.append({
                'feature_set': feature_set_name,
                'classifier': clf_name,
                'cv_repeat': cv_num,
                'stability_Lc_mean': patient_lc.mean(), 'stability_Lc_std': patient_lc.std(),
                'flip_count_mean': patient_flip.mean(), 'flip_count_std': patient_flip.std(),
                'n_patients': len(score_df),
            })
            for patient_id in score_df.index:
                patient_rows.append({
                    'feature_set': feature_set_name, 'classifier': clf_name, 'cv_repeat': cv_num,
                    'patient_id': patient_id,
                    'stability_Lc': patient_lc[patient_id], 'flip_count': patient_flip[patient_id],
                })

    # Keep one representative (classifier, cv_repeat) trajectory table around so
    # main() can plot a few example patients, mirroring CAREBench's Fig. 2.
    example_clf = 'XGBoost' if 'XGBoost' in trajectories else next(iter(trajectories))
    example_score_df, _ = trajectories[example_clf][0]

    result = (pd.DataFrame(summary_rows), pd.DataFrame(patient_rows), example_score_df, example_clf)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(done_path, 'wb') as f:
        pickle.dump(result, f)
    return result


def plot_example_trajectories(score_df, feature_set_name, clf_name, out_path, n_patients=5):
    rng = np.random.RandomState(42)
    patient_ids = rng.choice(score_df.index, size=min(n_patients, len(score_df)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for patient_id in patient_ids:
        row = score_df.loc[patient_id].dropna().sort_index()
        ax.plot(row.index, row.values, marker='o', markersize=3, label=str(patient_id))
    ax.set_xlabel('Test-time window (hours of monitoring available)')
    ax.set_ylabel('Risk score')
    ax.set_title(f'Example risk-score trajectories: {feature_set_name} / {clf_name}')
    ax.legend(fontsize=8, title='Patient ID')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lc_by_feature_set(patient_df, out_path):
    """Boxplot of per-patient Lc (pooled across all CV-repeats and patients) for
    each feature set, with classifiers shown as side-by-side boxes."""
    feature_sets = list(FEATURE_SETS.keys())
    clf_names = list(CLASSIFIERS.keys())
    width = 0.8 / len(clf_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for c, clf_name in enumerate(clf_names):
        offset = (c - (len(clf_names) - 1) / 2) * width
        positions = [i + offset for i in range(len(feature_sets))]
        data = [
            patient_df.loc[(patient_df['feature_set'] == fs) & (patient_df['classifier'] == clf_name),
                            'stability_Lc'].dropna()
            for fs in feature_sets
        ]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                        showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=colors[c % len(colors)], alpha=0.6)
        ax.plot([], [], color=colors[c % len(colors)], label=clf_name, linewidth=8, alpha=0.6)

    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets, rotation=15)
    ax.set_ylabel('Stability $L_c$ (lower = more stable)')
    ax.set_title('Per-patient stability by feature set (pooled across CV-repeats)')
    ax.legend(title='Classifier')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    all_summaries, all_patient_rows = [], []
    example_plotted = False

    for feature_set_name, feature_list in FEATURE_SETS.items():
        summary_df, patient_df, example_score_df, example_clf = run_feature_set(feature_set_name, feature_list)
        all_summaries.append(summary_df)
        all_patient_rows.append(patient_df)

        if not example_plotted:
            plot_example_trajectories(example_score_df, feature_set_name, example_clf,
                                       '../data/stability_score_example_trajectories.png')
            example_plotted = True

    combined = pd.concat(all_summaries, ignore_index=True)

    patient_level = pd.concat(all_patient_rows, ignore_index=True)
    patient_level.to_csv('../data/stability_score_per_patient.csv', index=False)

    plot_lc_by_feature_set(patient_level, '../data/stability_score_lc_by_feature_set.png')

    # Averaged-across-repeats view: mean of each repeat's mean, plus how much that
    # mean itself varies repeat-to-repeat (repeat-partition noise, distinct from
    # stability_Lc_std/flip_count_std in the by-repeat sheet, which is variability
    # across patients within a single repeat).
    summary = combined.groupby(['feature_set', 'classifier']).agg(
        stability_Lc_mean=('stability_Lc_mean', 'mean'),
        stability_Lc_std_across_repeats=('stability_Lc_mean', 'std'),
        flip_count_mean=('flip_count_mean', 'mean'),
        flip_count_std_across_repeats=('flip_count_mean', 'std'),
        n_repeats=('cv_repeat', 'nunique'),
    ).reset_index()

    excel_path = '../data/stability_score_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='By CV Repeat', index=False)
        summary.to_excel(writer, sheet_name='Summary (Averaged)', index=False)

    pivot_lc = summary.pivot_table(index='feature_set', columns='classifier', values='stability_Lc_mean')
    pivot_flip = summary.pivot_table(index='feature_set', columns='classifier', values='flip_count_mean')

    print(f"\n\n{'=' * 80}\nStability (local-Lipschitz Lc, lower = more stable)\n{'=' * 80}")
    print(pivot_lc.round(4).to_string())
    print(f"\n{'=' * 80}\nFlip count (lower = fewer alert transitions)\n{'=' * 80}")
    print(pivot_flip.round(3).to_string())

    print(f"\nFull results written to {excel_path} "
          f"('By CV Repeat' = all 160 rows, 'Summary (Averaged)' = collapsed across repeats)")
    print("Per-patient values written to ../data/stability_score_per_patient.csv")
    print("Example trajectories plotted to ../data/stability_score_example_trajectories.png")
    print("Lc-by-feature-set boxplot written to ../data/stability_score_lc_by_feature_set.png")

    # Run completed successfully and the real results are safely on disk -- clear
    # checkpoints so a future rerun (e.g. after changing the metric logic) doesn't
    # silently load stale cached results instead of recomputing.
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"Checkpoints in {CHECKPOINT_DIR} cleared (run completed successfully)")


if __name__ == "__main__":
    main()
