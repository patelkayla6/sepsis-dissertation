import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from dataset_preprocess import DataPreprocessor
from utils import set_seed, sofa_feature_names
from vif_analysis import (compute_vif, SCORE_COMPONENT_FEATURES, get_score_correlation,
                           print_score_correlation, plot_score_correlation)

# With only 158 patients / 22 deaths, the full ~67-feature clinical set (comorbidities,
# surgery flags, rare infection sources) causes perfect separation -- many of those flags
# have single-digit patient counts that share one outcome value, so MLE diverges.
# Scoped down to AGE plus the physiological variables that directly feed SOFA/NEWS
# (see SCORE_COMPONENT_FEATURES in vif_analysis.py): this both converges and gives the
# cleanest test of "does SOFA/NEWS add anything beyond its own known ingredients".
CORE_PHYSIOLOGY_FEATURES = SCORE_COMPONENT_FEATURES + ['AGE']
CORE_PHYSIOLOGY_PLUS_SCORES_FEATURES = CORE_PHYSIOLOGY_FEATURES + sofa_feature_names


def get_design_matrix_and_labels(feature_list, label_name='DIED'):
    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median',
                                     normalize=True, feature_list=feature_list)
    preprocessor.split_data(method='full')
    X, _, y, _ = preprocessor.get_dataset()
    return X, y.astype(int)


def clean_design_matrix(X):
    """Drop any exact-duplicate encodings so the design matrix isn't singular
    for MLE (these are exactly the VIF = inf features found in vif_analysis.py)."""
    X = X.astype(float)
    X = X.loc[:, X.std(axis=0) > 1e-8]

    corr = X.corr().abs()
    cols = corr.columns.tolist()
    to_drop = []
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] > 0.999:
                to_drop.append(c2)
    if to_drop:
        print(f"Dropping exact-duplicate columns: {to_drop}")
        X = X.drop(columns=to_drop)
    return X


def build_cleaned_design_matrix(feature_list, label_name='DIED'):
    """Fetch + clean the design matrix once. Used to build the superset
    (physiology + SOFA/NEWS) matrix so the reduced model can be derived by
    dropping columns from it directly, guaranteeing the two models are fit
    on identical rows and strictly nested columns -- required for the LRT
    below to be valid."""
    X, y = get_design_matrix_and_labels(feature_list, label_name)
    return clean_design_matrix(X), y


def fit_logistic_regression(X, y):
    X_const = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X_const).fit(disp=0, maxiter=200)
    if not model.mle_retvals.get('converged', True):
        print("WARNING: MLE did not converge -- coefficients/p-values below may be unreliable.")
    return model


def summarize_model(model, highlight_features=None):
    results = pd.DataFrame({
        'coef': model.params,
        'std_err': model.bse,
        'z': model.tvalues,
        'p_value': model.pvalues,
    })
    results['significant_p<0.05'] = results['p_value'] < 0.05
    print(results.round(4).to_string())

    if highlight_features:
        present = [f for f in highlight_features if f in results.index]
        if present:
            print("\nSOFA/NEWS terms:")
            print(results.loc[present].round(4).to_string())

    return results


def run_model(name, X, y, highlight_features=None, show_correlation=False):
    print(f"\n{'='*70}\nModel: {name}\n{'='*70}")

    vif_df = compute_vif(X)
    flagged = vif_df[vif_df['VIF'] > 5]
    print(f"VIF check ({X.shape[1]} features): {len(flagged)} feature(s) with VIF > 5")
    print(vif_df.to_string(index=False))

    if show_correlation:
        print_score_correlation(X, component_features=SCORE_COMPONENT_FEATURES)
        plot_score_correlation(X, component_features=SCORE_COMPONENT_FEATURES,
                                save_path='clinical_significance_score_correlation.png')

    model = fit_logistic_regression(X, y)
    print(f"\nLogistic regression (n={len(y)}, events={int(y.sum())}):")
    coef_df = summarize_model(model, highlight_features)
    return model, vif_df, coef_df


def likelihood_ratio_test(model_full, model_reduced, name_full='full', name_reduced='reduced'):
    """Chi-square LRT for two nested logistic models. Requires model_reduced's
    design matrix to be a strict column subset of model_full's, fit on the
    same rows -- see build_cleaned_design_matrix."""
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df_diff = int(round(model_full.df_model - model_reduced.df_model))
    p_value = chi2.sf(lr_stat, df_diff)
    print(f"\nLikelihood ratio test ({name_full} vs. {name_reduced}):")
    print(f"  LR stat = {lr_stat:.4f}, df = {df_diff}, p = {p_value:.4g}")
    return lr_stat, df_diff, p_value


def main():
    set_seed(42)
    label_name = 'DIED'

    X_full, y = build_cleaned_design_matrix(CORE_PHYSIOLOGY_PLUS_SCORES_FEATURES, label_name)
    score_cols_present = [c for c in sofa_feature_names if c in X_full.columns]
    X_reduced = X_full.drop(columns=score_cols_present)

    model_reduced, vif_reduced, coef_reduced = run_model('physiology_only', X_reduced, y)
    model_full, vif_full, coef_full = run_model(
        'physiology_plus_SOFA_NEWS', X_full, y,
        highlight_features=sofa_feature_names, show_correlation=True)

    lr_stat, df_diff, p_value = likelihood_ratio_test(
        model_full, model_reduced,
        name_full='physiology_plus_SOFA_NEWS', name_reduced='physiology_only')

    vif_all = pd.concat([
        vif_reduced.assign(model='physiology_only'),
        vif_full.assign(model='physiology_plus_SOFA_NEWS'),
    ], ignore_index=True)
    vif_all.to_csv('../data/clinical_significance_vif.csv', index=False)

    coef_all = pd.concat([
        coef_reduced.rename_axis('feature').reset_index().assign(model='physiology_only'),
        coef_full.rename_axis('feature').reset_index().assign(model='physiology_plus_SOFA_NEWS'),
    ], ignore_index=True)
    coef_all.to_csv('../data/clinical_significance_coefficients.csv', index=False)

    corr_df = get_score_correlation(X_full, component_features=SCORE_COMPONENT_FEATURES)
    corr_df.rename_axis('component_feature').to_csv('../data/clinical_significance_score_correlation.csv')

    lrt_df = pd.DataFrame([{
        'model_full': 'physiology_plus_SOFA_NEWS',
        'model_reduced': 'physiology_only',
        'lr_stat': lr_stat,
        'df': df_diff,
        'p_value': p_value,
    }])
    lrt_df.to_csv('../data/clinical_significance_lrt.csv', index=False)

    print("\nSaved: ../data/clinical_significance_vif.csv, ../data/clinical_significance_coefficients.csv, "
          "../data/clinical_significance_score_correlation.csv, ../data/clinical_significance_lrt.csv")


if __name__ == '__main__':
    main()
