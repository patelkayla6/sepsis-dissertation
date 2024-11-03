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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.decomposition import PCA
import warnings
from dataset_preprocess import DataPreprocessor
from utils import set_seed
from scipy.stats import wilcoxon
import copy
from itertools import combinations
import shap

# Ignore warnings
warnings.filterwarnings("ignore")

# Function to compute evaluation metrics
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

def train_and_evaluate_classifier(clf_name, clf, label_name, calibrator='none', split_method='kfold', kfold=5, num_splits=50, use_pca=False, n_components=None, feature_list=None):
    print(f"\nClassifier: {clf_name} with Calibrator: {calibrator} {'with PCA' if use_pca else ''}")
    metrics_list = []
    auroc_list = []
    model_list = []
    feature_names = None  # Initialize feature names
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median', normalize=True, feature_list=feature_list)
    
    assert split_method in ['kfold', 'leaveoneout']
    if split_method == 'leaveoneout':
        num_splits = preprocessor.get_number_of_samples()
        # Initialize lists to store y_true, y_prob, y_pred for LOOCV
        y_true_list = []
        y_prob_list = []
        y_pred_list = []
        
        for split_index in range(num_splits):
            # Split the data
            preprocessor.split_data(method='leaveoneout', split_index=split_index)
            X_train, X_test, y_train, y_test = preprocessor.get_dataset()
            
            # Ensure the labels are integers
            y_train, y_test = y_train.astype(int), y_test.astype(int)
            
            # Apply PCA if requested
            if use_pca:
                pca = PCA(n_components=n_components)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                # Update feature names to principal components
                feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
            else:
                # Get feature names
                if feature_names is None:
                    if isinstance(X_train, pd.DataFrame):
                        feature_names = X_train.columns.tolist()
                    else:
                        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
            
            # Fit the classifier
            clf.fit(X_train, y_train)
            
            # Get classifier outputs on training data
            if hasattr(clf, "predict_proba"):
                y_prob_train = clf.predict_proba(X_train)[:, 1]
            else:  # For classifiers that do not have predict_proba (e.g., SVM)
                y_prob_train = clf.decision_function(X_train)
            
            # Initialize calibrator
            if calibrator == 'platt':
                calib_model = PlattScaling()
                calib_model.fit(y_prob_train.reshape(-1, 1), y_train)
                # Calibrated probabilities on training data
                y_prob_train_cal = calib_model.predict_proba(y_prob_train.reshape(-1, 1))[:, 1]
            elif calibrator == 'isotonic':
                calib_model = IsotonicRegression(out_of_bounds='clip')
                calib_model.fit(y_prob_train, y_train)
                y_prob_train_cal = calib_model.predict(y_prob_train)
            else:  # 'none'
                y_prob_train_cal = y_prob_train
            
            # Compute optimal threshold on calibrated training data
            fpr, tpr, thresholds = roc_curve(y_train, y_prob_train_cal)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Get classifier outputs on test data
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.decision_function(X_test)
            
            # Apply calibrator to test data
            if calibrator == 'platt':
                y_prob_cal = calib_model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
            elif calibrator == 'isotonic':
                y_prob_cal = calib_model.predict(y_prob)
            else:  # 'none'
                y_prob_cal = y_prob
            
            # Make predictions using the optimal threshold
            y_pred = (y_prob_cal >= optimal_threshold).astype(int)
            
            # Store y_test, y_prob_cal, y_pred
            y_true_list.append(y_test)
            y_prob_list.append(y_prob_cal)
            y_pred_list.append(y_pred)
        
        # After all splits
        # Concatenate y_true_list, y_prob_list, y_pred_list
        y_true_all = np.concatenate(y_true_list)
        y_prob_all = np.concatenate(y_prob_list)
        y_pred_all = np.concatenate(y_pred_list)
        
        # Compute metrics
        metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
        metrics_list.append(metrics)
        auroc_list.append(metrics['AUROC'])
        
        # Print metrics
        print("Metrics over Leave-One-Out Cross-Validation:")
        print(pd.Series(metrics))
        
        return metrics_list, auroc_list, model_list, feature_names
    
    if split_method == 'kfold':
        num_CVs = num_splits // kfold
        
        for cv_num in range(num_CVs):
            # Initialize lists to store y_true, y_prob, y_pred for this CV
            y_true_cv = []
            y_prob_cv = []
            y_pred_cv = []
            
            for fold_num in range(kfold):
                # Calculate split_index
                split_index = cv_num * kfold + fold_num
                
                # Split the data
                preprocessor.split_data(method='kfold', n_splits=kfold, split_index=split_index)
                X_train, X_test, y_train, y_test = preprocessor.get_dataset()
                
                # Ensure the labels are integers
                y_train, y_test = y_train.astype(int), y_test.astype(int)
                
                # Apply PCA if requested
                if use_pca:
                    pca = PCA(n_components=n_components)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                    # Update feature names to principal components
                    feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
                else:
                    # Get feature names
                    if feature_names is None:
                        if isinstance(X_train, pd.DataFrame):
                            feature_names = X_train.columns.tolist()
                        else:
                            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
                
                # Fit the classifier
                clf.fit(X_train, y_train)
                
                # Get classifier outputs on training data
                if hasattr(clf, "predict_proba"):
                    y_prob_train = clf.predict_proba(X_train)[:, 1]
                else:  # For classifiers that do not have predict_proba (e.g., SVM)
                    y_prob_train = clf.decision_function(X_train)
                
                # Initialize calibrator
                if calibrator == 'platt':
                    calib_model = PlattScaling()
                    calib_model.fit(y_prob_train.reshape(-1, 1), y_train)
                    # Calibrated probabilities on training data
                    y_prob_train_cal = calib_model.predict_proba(y_prob_train.reshape(-1, 1))[:, 1]
                elif calibrator == 'isotonic':
                    calib_model = IsotonicRegression(out_of_bounds='clip')
                    calib_model.fit(y_prob_train, y_train)
                    y_prob_train_cal = calib_model.predict(y_prob_train)
                else:  # 'none'
                    y_prob_train_cal = y_prob_train
                
                # Compute optimal threshold on calibrated training data
                fpr, tpr, thresholds = roc_curve(y_train, y_prob_train_cal)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                # Get classifier outputs on test data
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                else:
                    y_prob = clf.decision_function(X_test)
                
                # Apply calibrator to test data
                if calibrator == 'platt':
                    y_prob_cal = calib_model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
                elif calibrator == 'isotonic':
                    y_prob_cal = calib_model.predict(y_prob)
                else:  # 'none'
                    y_prob_cal = y_prob
                
                # Make predictions using the optimal threshold
                y_pred = (y_prob_cal >= optimal_threshold).astype(int)
                
                # Store y_test, y_prob_cal, y_pred
                y_true_cv.append(y_test)
                y_prob_cv.append(y_prob_cal)
                y_pred_cv.append(y_pred)
            
            # After all folds in this CV
            # Concatenate y_true_cv, y_prob_cv, y_pred_cv
            y_true_cv = np.concatenate(y_true_cv)
            y_prob_cv = np.concatenate(y_prob_cv)
            y_pred_cv = np.concatenate(y_pred_cv)
            
            # Compute metrics
            metrics = compute_metrics(y_true_cv, y_pred_cv, y_prob_cv)
            metrics_list.append(metrics)
            
            # Store AUROC score
            auroc_list.append(metrics['AUROC'])
        
        # Compute average metrics and standard deviation across CVs
        avg_metrics = pd.DataFrame(metrics_list).mean()
        std_metrics = pd.DataFrame(metrics_list).std()

        # Convert to percentages and round to one decimal place
        avg_metrics_percent = (avg_metrics * 100).round(1)
        std_metrics_percent = (std_metrics * 100).round(1)

        # Combine mean and std into a single Series with formatted strings
        combined_metrics = avg_metrics_percent.astype(str) + '%±' + std_metrics_percent.astype(str) + '%'

        print("Average Metrics over {} {}-fold cross-validations (mean±std):".format(num_CVs, kfold))
        print(combined_metrics)
        
        return metrics_list, auroc_list, model_list, feature_names

def perform_statistical_tests(auroc_scores, classifiers_list):
    # Perform statistical tests between classifiers
    auroc_df = pd.DataFrame(auroc_scores)
    p_values = pd.DataFrame(index=classifiers_list, columns=classifiers_list)
    
    print("\nP-values between classifiers based on AUROC scores:")
    for clf1, clf2 in combinations(classifiers_list, 2):
        # Perform Wilcoxon signed-rank test
        stat, p = wilcoxon(auroc_df[clf1], auroc_df[clf2])
        p_values.loc[clf1, clf2] = p
        p_values.loc[clf2, clf1] = p  # Since it's symmetric
        print(f"P-value between {clf1} and {clf2}: {p:.4f}")
    
    # For display purposes, fill diagonal with NaN
    np.fill_diagonal(p_values.values, np.nan)
    print("\nP-values matrix:")
    print(p_values)
    
    return p_values

def compute_feature_importance(clf_name, clf, preprocessor, feature_names, use_pca=False, n_components=None):
    # Train the classifier on the whole dataset
    preprocessor.split_data(method='full')
    X_train, _, y_train, _ = preprocessor.get_dataset()
    y_train = y_train.astype(int)
    
    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        # Update feature names to principal components
        feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
        # # print explained_variance_ratio_
        # print(pca.explained_variance_ratio_)
        # print([sum(pca.explained_variance_ratio_[:i]) for i in range(1, 50)])
    
    # Fit the classifier on the whole dataset
    clf.fit(X_train, y_train)
    
    print(f"\nTop 20 Most Important Features for {clf_name} trained on the whole dataset{' with PCA' if use_pca else ''}:")
    # Get feature importances
    if clf_name.startswith('Logistic_Regression'):
        importances = np.abs(clf.coef_[0])
    elif clf_name.startswith('Random_Forest'):
        importances = clf.feature_importances_
    elif clf_name.startswith('XGBoost'):
        importances = clf.feature_importances_
    else:
        print(f"No feature importance available for classifier {clf_name}.")
        return
    # Create DataFrame of feature importances
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print(importances_df.head(20).to_string(index=False))


def compute_shap_feature_importance(clf_name, clf, preprocessor, feature_names, use_pca=False, n_components=None):
    # Train the classifier on the whole dataset
    preprocessor.split_data(method='full')
    X_train, _, y_train, _ = preprocessor.get_dataset()
    y_train = y_train.astype(int)
    
    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        # Update feature names to principal components
        feature_names = [f'PC_{i+1}' for i in range(X_train.shape[1])]
    
    # Fit the classifier on the whole dataset
    clf.fit(X_train, y_train)
    
    print(f"\nTop 20 Most Important Features for {clf_name} using SHAP values trained on the whole dataset{' with PCA' if use_pca else ''}:")
    
    # Compute SHAP values
    try:
        if clf_name.startswith('Logistic_Regression'):
            # Use LinearExplainer for linear models
            explainer = shap.LinearExplainer(clf, X_train)
            shap_values = explainer.shap_values(X_train)
        elif clf_name.startswith('Random_Forest') or clf_name.startswith('XGBoost'):
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)
        elif clf_name.startswith('SVM'):
            # SHAP computation for SVM using KernelExplainer can be computationally intensive
            # For demonstration, we will skip it or you can sample a subset
            print(f"SHAP value computation is computationally intensive for {clf_name}. Skipping.")
            return
        else:
            print(f"No SHAP explainer available for classifier {clf_name}.")
            return
    except Exception as e:
        print(f"Error computing SHAP values for {clf_name}: {e}")
        return
    
    # For binary classification, shap_values can be a numpy array
    # For multiclass, it's a list of arrays
    if isinstance(shap_values, list):
        # For binary classification, shap_values[1] corresponds to the positive class
        shap_values = shap_values[1]
    
    # Aggregate the absolute SHAP values over the dataset to get feature importance
    importances = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame of feature importances
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print(importances_df.head(20).to_string(index=False))


def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Define the label name (e.g., 'DIED', 'HFNOV', 'MEVEN')
    label_name = 'DIED'
    print(f"\nLabel: {label_name}")
    
    # Define classifiers
    classifiers = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Calibrator options
    # calibrators = ['none', 'platt', 'isotonic']
    calibrators = ['none']
    
    # PCA options
    use_pca = False  # Set to True to enable PCA
    n_components = 50  # Number of principal components to keep
    
    # Dictionaries to store results
    auroc_scores = {}
    models_info = {}
    feature_names = None  # Initialize feature names
    
    # Split method
    split_method = 'kfold'  # Set to 'leaveoneout' for Leave-One-Out CV
    # K for k-fold (not used in leave-one-out)
    kfold = 10
    # Number of splits (will be set inside the function for leave-one-out)
    num_splits = 200  # Set to None since it will be determined by the number of samples

    # Set feature list to None to use all features
    ppg_features = [
        'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
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
        'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC', 'PAV'
    ]
    feature_type = ['mean', 'var', 'slope', 'intercept', 'rmse', 'r2']
    feature_list = [feature+'_'+ftype for feature in ppg_features for ftype in feature_type]
    
    # Loop over classifiers and calibrators
    for calibrator in calibrators:
        for clf_name, clf in classifiers.items():
            combined_name = f"{clf_name}_{calibrator}{'_PCA' if use_pca else ''}"
            metrics_list, auroc_list, model_list, feature_names = train_and_evaluate_classifier(
                clf_name, copy.deepcopy(clf), label_name, calibrator, split_method, kfold, num_splits, use_pca, n_components, feature_list)
            # Store AUROC scores and model information
            auroc_scores[combined_name] = auroc_list
            models_info[combined_name] = model_list
    
    # Perform statistical tests between classifiers
    classifiers_list = list(auroc_scores.keys())
    p_values = perform_statistical_tests(auroc_scores, classifiers_list)
    
    # Compute feature importances using the classifier trained on the whole dataset
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(label_name=label_name, missing_method='median', normalize=True, feature_list=feature_list)
    preprocessor.split_data(method='full')
    X_full, _, y_full, _ = preprocessor.get_dataset()
    y_full = y_full.astype(int)
    if isinstance(X_full, pd.DataFrame):
        feature_names_full = X_full.columns.tolist()
    else:
        feature_names_full = [f'Feature_{i}' for i in range(X_full.shape[1])]
    
    for clf_name, clf in classifiers.items():
        if clf_name in ['Logistic_Regression', 'Random_Forest', 'XGBoost']:
            compute_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, feature_names_full, use_pca, n_components)


    # Compute feature importances using SHAP values
    for clf_name, clf in classifiers.items():
        compute_shap_feature_importance(clf_name, copy.deepcopy(clf), preprocessor, feature_names_full, use_pca, n_components)

    
if __name__ == "__main__":
    main()
