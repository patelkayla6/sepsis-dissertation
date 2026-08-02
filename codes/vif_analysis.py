import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from dataset_preprocess import DataPreprocessor
from utils import set_seed, enrol_feature_names, vital_sign_feature_names, sofa_feature_names

# dataset_preprocess.py unconditionally computes mean/var over PPG/ECG HRV features for
# every patient even though this module never uses them, which floods stdout with
# "Mean of empty slice" / "Degrees of freedom <= 0" RuntimeWarnings on patients with
# missing/short PPG data. Not relevant to the clinical-feature VIF work done here.
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Clinical-only = demographics/comorbidities/admission vitals/labs + continuous monitoring vitals.
# Clinical + scores adds the derived SOFA/NEWS composite scores on top.
CLINICAL_ONLY_FEATURES = enrol_feature_names + vital_sign_feature_names
CLINICAL_PLUS_SCORES_FEATURES = CLINICAL_ONLY_FEATURES + sofa_feature_names

# Clinical variables that feed directly into the SOFA/NEWS composite scores,
# used to inspect exactly which raw inputs each score is redundant with.
SCORE_COMPONENT_FEATURES = [
    'ENR_GCS', 'ENR_SPO2', 'ENR_RESP', 'ENR_SYSBP', 'MONITORING_MAP_mean',
    'ENR_CREAT', 'ENR_PLAT', 'ENR_TEMP', 'ENR_PULSE',
]


def get_design_matrix(feature_list, label_name='DIED'):
    """Full-data, imputed, standardized design matrix (one-hot encoded) for VIF computation."""
    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median',
                                     normalize=True, feature_list=feature_list)
    preprocessor.split_data(method='full')
    X_train, _, _, _ = preprocessor.get_dataset()
    return X_train


def compute_vif(X):
    """Variance Inflation Factor per feature. VIF > 5 is commonly flagged, > 10 is severe."""
    X = X.astype(float)  # one-hot columns come back as bool, which breaks the OLS fit below
    X = X.loc[:, X.std(axis=0) > 1e-8].copy()  # drop constant columns, they break the OLS fit
    X_const = add_constant(X, has_constant='add')
    records = []
    for i, col in enumerate(X_const.columns):
        if col == 'const':
            continue
        vif = variance_inflation_factor(X_const.values, i)
        records.append({'Feature': col, 'VIF': vif})
    return pd.DataFrame(records).sort_values('VIF', ascending=False).reset_index(drop=True)


def run_vif_check(label_name='DIED'):
    feature_sets = {
        'clinical_only': CLINICAL_ONLY_FEATURES,
        'clinical_plus_SOFA_NEWS': CLINICAL_PLUS_SCORES_FEATURES,
    }
    for name, feats in feature_sets.items():
        X = get_design_matrix(feats, label_name)
        vif_df = compute_vif(X)
        print(f"\n=== VIF for '{name}' ({X.shape[1]} encoded features) ===")
        print(vif_df.to_string(index=False))
        flagged = vif_df[vif_df['VIF'] > 5]
        print(f"\n{len(flagged)} feature(s) with VIF > 5:")
        print(flagged.to_string(index=False))


def get_score_correlation(X, component_features=SCORE_COMPONENT_FEATURES, score_features=None):
    """Pairwise Pearson correlation between SOFA/NEWS scores and the raw clinical
    variables that feed into them (rows=components, columns=scores), or None if
    neither side is present in X."""
    score_features = sofa_feature_names if score_features is None else score_features
    components = [c for c in component_features if c in X.columns]
    scores = [c for c in score_features if c in X.columns]
    if not components or not scores:
        return None
    return X[components + scores].corr().loc[components, scores]


def print_score_correlation(X, component_features=SCORE_COMPONENT_FEATURES, score_features=None):
    corr = get_score_correlation(X, component_features, score_features)
    if corr is None:
        return
    print("\n=== Correlation: SOFA/NEWS scores vs. their component clinical variables ===")
    print(corr.round(3).to_string())


def plot_score_correlation(X, component_features=SCORE_COMPONENT_FEATURES, score_features=None,
                            save_path='score_correlation_heatmap.png'):
    """Heatmap of the same SOFA/NEWS vs. component-variable correlation table, annotated with values."""
    corr = get_score_correlation(X, component_features, score_features)
    if corr is None:
        return

    fig, ax = plt.subplots(figsize=(1.1 * len(corr.columns) + 2, 0.5 * len(corr.index) + 2))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > 0.5 else 'black')

    fig.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title('SOFA/NEWS scores vs. component clinical variables')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved correlation heatmap to {save_path}")
    plt.show()


def run_score_correlation_check(label_name='DIED'):
    X = get_design_matrix(CLINICAL_PLUS_SCORES_FEATURES, label_name)
    print_score_correlation(X)
    plot_score_correlation(X)


def main():
    set_seed(42)
    label_name = 'DIED'
    run_vif_check(label_name)
    run_score_correlation_check(label_name)


if __name__ == '__main__':
    main()
