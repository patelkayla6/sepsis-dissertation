import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
import warnings
from dataset_preprocess import DataPreprocessor
from utils import *
from scipy.stats import wilcoxon
import copy
from itertools import combinations
import shap

# Ignore warnings
warnings.filterwarnings("ignore")

def prepare_data(preprocessor, use_pca, n_components, feature_names):
    """
    Prepare training and testing data, apply PCA if requested, 
    and extract feature names.
    Now, if n_components < 1.0, it is treated as the fraction of 
    variance to retain; if n_components >= 1, it is treated as 
    the number of principal components.
    """
    X_train, X_test, y_train, y_test = preprocessor.get_dataset()
    y_train, y_test = y_train.astype(int), y_test.astype(int)

    # Apply PCA if requested
    if use_pca and n_components is not None:
        if n_components < 1.0:
            # Interpret n_components as the fraction of variance to retain
            pca = PCA(n_components=n_components, svd_solver='full')
        else:
            # Interpret n_components as the actual number of components
            pca = PCA(n_components=int(n_components))
        
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Renaming features to principal components
        feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
    else:
        # If not using PCA, just ensure we have a list of feature names
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

    return X_train, X_test, y_train, y_test, feature_names


def fit_and_calibrate_classifier(clf, X_train, y_train, calibrator):
    """Fit the classifier and calibrate probabilities if requested."""
    clf.fit(X_train, y_train)
    y_prob_train = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_train)

    if calibrator == 'platt':
        calib_model = PlattScaling()
        calib_model.fit(y_prob_train.reshape(-1, 1), y_train)
        y_prob_train_cal = calib_model.predict_proba(y_prob_train.reshape(-1, 1))[:, 1]
    elif calibrator == 'isotonic':
        calib_model = IsotonicRegression(out_of_bounds='clip')
        calib_model.fit(y_prob_train, y_train)
        y_prob_train_cal = calib_model.predict(y_prob_train)
    else:
        calib_model = None
        y_prob_train_cal = y_prob_train

    return clf, calib_model, y_prob_train_cal

def compute_optimal_threshold(y_train, y_prob_train_cal):
    """Compute the optimal threshold based on the training data."""
    fpr, tpr, thresholds = roc_curve(y_train, y_prob_train_cal)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def predict_with_calibrator(clf, calib_model, X_test, calibrator):
    """Predict probabilities on test data and apply calibration."""
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    if calibrator == 'platt':
        y_prob_cal = calib_model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    elif calibrator == 'isotonic':
        y_prob_cal = calib_model.predict(y_prob)
    else:
        y_prob_cal = y_prob
    return y_prob_cal

def process_single_split(preprocessor, clf, calibrator, use_pca, n_components, feature_names, use_RFE=False, RFE_feature_number=None):
    """Process a single data split: train, optionally perform RFE, calibrate, and predict."""
    X_train, X_test, y_train, y_test, feature_names = prepare_data(preprocessor, use_pca, n_components, feature_names)

    if use_RFE:
        if RFE_feature_number is None:
            raise ValueError("RFE_feature_number must be provided when use_RFE is True.")
        
        # Apply RFE to select features
        rfe = RFE(estimator=clf, n_features_to_select=RFE_feature_number, step=1)
        rfe.fit(X_train, y_train)
        
        # Transform training and test sets
        X_train = rfe.transform(X_train)
        X_test = rfe.transform(X_test)

        # Update feature names based on RFE selection
        if feature_names is not None:
            feature_names = [f for f, s in zip(feature_names, rfe.support_) if s]

    # Fit and calibrate classifier
    clf, calib_model, y_prob_train_cal = fit_and_calibrate_classifier(clf, X_train, y_train, calibrator)

    # Compute the optimal threshold on the training set
    optimal_threshold = compute_optimal_threshold(y_train, y_prob_train_cal)

    # Predict on test set using the calibrator
    y_prob_cal = predict_with_calibrator(clf, calib_model, X_test, calibrator)
    y_pred = (y_prob_cal >= optimal_threshold).astype(int)

    return y_test, y_prob_cal, y_pred, feature_names


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    return {
        'AUROC': auroc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1-score': f1,
        'AUPRC': auprc
    }

def train_and_evaluate_classifier(clf_name, clf, label_name, calibrator='none', split_method='kfold', kfold=5,
                                  num_splits=50, use_pca=False, n_components=None, feature_list=None, 
                                  use_RFE=False, RFE_feature_number=None):
    """Train and evaluate the classifier using specified cross-validation method."""
    print(f"\nClassifier: {clf_name} "
          f"{f'with calibrator {calibrator}' if calibrator!='none' else ''} "
          f"{'with PCA' if use_pca else ''} "
          f"{'with RFE' if use_RFE else ''}{f' (selecting {RFE_feature_number} features)' if use_RFE else ''}")

    metrics_list = []
    auroc_list, auprc_list = [], []
    all_y_test, all_y_prob = [], []
    feature_names = None

    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median', normalize=True,
                                    feature_list=feature_list)
    assert calibrator in ['none', 'platt', 'isotonic']
    assert split_method in ['kfold', 'leaveoneout']

    if split_method == 'leaveoneout':
        num_splits = preprocessor.get_number_of_samples()
        splits = range(num_splits)
        y_true_list, y_prob_list, y_pred_list = [], [], []

        for split_index in splits:
            preprocessor.split_data(method='leaveoneout', split_index=split_index)
            y_test, y_prob_cal, y_pred, feature_names = process_single_split(preprocessor, copy.deepcopy(clf),
                                                                             calibrator, use_pca, n_components,
                                                                             feature_names, use_RFE, RFE_feature_number)
            y_true_list.append(y_test)
            y_prob_list.append(y_prob_cal)
            y_pred_list.append(y_pred)

        y_true_all = np.concatenate(y_true_list)
        y_prob_all = np.concatenate(y_prob_list)
        y_pred_all = np.concatenate(y_pred_list)

        metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
        metrics_list.append(metrics)
        auroc_list.append(metrics['AUROC'])
        auprc_list.append(metrics['AUPRC'])

        print("Metrics over Leave-One-Out Cross-Validation:")
        print(pd.Series(metrics))

    else:  # kfold
        num_CVs = num_splits // kfold

        for cv_num in range(num_CVs):
            y_true_cv, y_prob_cv, y_pred_cv = [], [], []

            for fold_num in range(kfold):
                split_index = cv_num * kfold + fold_num
                preprocessor.split_data(method='kfold', n_splits=kfold, split_index=split_index)
                y_test, y_prob_cal, y_pred, feature_names = process_single_split(preprocessor, copy.deepcopy(clf),
                                                                                 calibrator, use_pca, n_components,
                                                                                 feature_names, use_RFE, RFE_feature_number)
                y_true_cv.append(y_test)
                y_prob_cv.append(y_prob_cal)
                y_pred_cv.append(y_pred)
                # Also store them in the overall lists to later plot the ROC
                all_y_test.append(y_test)
                all_y_prob.append(y_prob_cal)

            y_true_cv = np.concatenate(y_true_cv)
            y_prob_cv = np.concatenate(y_prob_cv)
            y_pred_cv = np.concatenate(y_pred_cv)

            metrics = compute_metrics(y_true_cv, y_pred_cv, y_prob_cv)
            metrics_list.append(metrics)
            auroc_list.append(metrics['AUROC'])
            auprc_list.append(metrics['AUPRC'])

        avg_metrics = pd.DataFrame(metrics_list).mean()
        std_metrics = pd.DataFrame(metrics_list).std()

        avg_metrics_percent = (avg_metrics * 100).round(1)
        std_metrics_percent = (std_metrics * 100).round(1)
        combined_metrics = avg_metrics_percent.astype(str) + '%±' + std_metrics_percent.astype(str) + '%'

        print(f"Average Metrics over {num_CVs} {kfold}-fold cross-validations (mean±std):")
        print(combined_metrics)

    return metrics_list, auroc_list, auprc_list, feature_names, np.concatenate(all_y_test), np.concatenate(all_y_prob)

def perform_statistical_tests(scores, classifiers_list, metric='AUROC'):
    """Perform statistical tests between classifiers based on scores."""
    df = pd.DataFrame(scores)
    p_values = pd.DataFrame(index=classifiers_list, columns=classifiers_list)

    print(f"\nP-values between classifiers based on {metric} scores:")
    for clf1, clf2 in combinations(classifiers_list, 2):
        stat, p = wilcoxon(df[clf1], df[clf2])
        p_values.loc[clf1, clf2] = p
        p_values.loc[clf2, clf1] = p
        print(f"P-value between {clf1} and {clf2}: {p:.4f}")

    np.fill_diagonal(p_values.values, np.nan)
    print("\nP-values matrix:")
    print(p_values)

    return p_values

def train_classifier_on_full_data(clf_name, clf, preprocessor, use_pca=False, n_components=None):
    """Train the classifier on the full dataset."""
    preprocessor.split_data(method='full')
    X_train, _, y_train, _ = preprocessor.get_dataset()
    y_train = y_train.astype(int)

    if use_pca and n_components is not None:
        # Same fraction-of-variance vs. number-of-components logic
        if n_components < 1.0:
            pca = PCA(n_components=n_components, svd_solver='full')
        else:
            pca = PCA(n_components=int(n_components))
        X_train = pca.fit_transform(X_train)
        feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
    else:
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

    clf.fit(X_train, y_train)
    return clf, feature_names, X_train, y_train


def compute_feature_importance(clf_name, clf, preprocessor, use_pca=False, n_components=None):
    """Compute and display feature importance based on the trained classifier."""
    clf, feature_names, _, _ = train_classifier_on_full_data(clf_name, clf, preprocessor, use_pca, n_components)
    print(f"\nTop 20 Most Important Features for {clf_name} trained on the whole dataset{' with PCA' if use_pca else ''}:")

    if clf_name.startswith('Logistic_Regression'):
        importances = np.abs(clf.coef_[0])
    elif clf_name.startswith('Random_Forest'):
        importances = clf.feature_importances_
    elif clf_name.startswith('XGBoost'):
        importances = clf.feature_importances_
    else:
        print(f"No feature importance available for classifier {clf_name}.")
        return

    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print(importances_df.head(20).to_string(index=False))

def compute_shap_feature_importance(clf_name, clf, preprocessor, use_pca=False, n_components=None):
    """Compute and display feature importance using SHAP values."""
    clf, feature_names, X_train, y_train = train_classifier_on_full_data(clf_name, clf, preprocessor, use_pca, n_components)
    print(f"\nTop 20 Most Important Features for {clf_name} using SHAP values trained on the whole dataset{' with PCA' if use_pca else ''}:")
    try:
        if clf_name.startswith('Logistic_Regression'):
            explainer = shap.LinearExplainer(clf, X_train)
            shap_values = explainer.shap_values(X_train)
        elif clf_name.startswith('Random_Forest') or clf_name.startswith('XGBoost'):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)
        else:
            print(f"No SHAP explainer available for classifier {clf_name}.")
            return
    except Exception as e:
        print(f"Error computing SHAP values for {clf_name}: {e}")
        return

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:,:,1]

    importances = np.abs(shap_values).mean(axis=0)

    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print(importances_df.head(20).to_string(index=False))

def main():
    set_seed(42)
    label_name = 'DIED'
    print(f"\nLabel: {label_name}")

    # classifiers = {
    #     'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    #     'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    #     'Random_Forest': RandomForestClassifier(random_state=42),
    #     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    # }
    classifiers = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    calibrators = ['none']
    use_pca = False
    n_components = 0.9

    auroc_scores, auprc_scores = {}, {}

    feature_list = normal_feature_names + PPG_feature_names + ECG_feature_names
    print('feature list', feature_list)
    print('normal feature name', normal_feature_names)
    print('ppg feature name', PPG_feature_names)
    print('ecg feature name', ECG_feature_names)

    split_method = 'kfold'
    kfold = 10
    num_splits = 200
    use_RFE = False
    RFE_feature_number=100

    for calibrator in calibrators:
        for clf_name, clf in classifiers.items():
            combined_name = f"{clf_name}_{calibrator}{'_PCA' if use_pca else ''}"
            metrics_list, auroc_list, auprc_list, feature_names, _, _ = train_and_evaluate_classifier(
                clf_name, copy.deepcopy(clf), label_name, calibrator, split_method, kfold, num_splits,
                use_pca, n_components, feature_list, use_RFE, RFE_feature_number)
            auroc_scores[combined_name] = auroc_list
            auprc_scores[combined_name] = auprc_list

    classifiers_list = list(auroc_scores.keys())
    p_values = perform_statistical_tests(auroc_scores, classifiers_list, "AUROC")
    p_values = perform_statistical_tests(auprc_scores, classifiers_list, "AUPRC")

    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median', normalize=True,
                                    feature_list=feature_list)

    for clf_name, clf in classifiers.items():
        if clf_name in ['Logistic_Regression', 'Random_Forest', 'XGBoost']:
            compute_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

    for clf_name, clf in classifiers.items():
        compute_shap_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, use_pca, n_components)

if __name__ == "__main__":
    main()
