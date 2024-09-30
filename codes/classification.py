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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
from dataset_preprocess import DataPreprocessor
from utils import set_seed
from scipy.stats import wilcoxon
import copy

# Ignore warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
set_seed(42)

# Define the label name (e.g., 'DIED', 'HFNOV', 'MEVEN')
label_name = 'HFNOV'
print(f"Label: {label_name}")

# Define classifiers
classifiers = {
    'Naive_Bayes': GaussianNB(),
    'K-Nearest_Neighbors': KNeighborsClassifier(),
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM_(RBF_Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random_Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Function to compute evaluation metrics
def compute_metrics(y_true, y_pred, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
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

# Dictionary to store AUROC scores for each classifier
auroc_scores = {}
# Dictionary to store models and their test AUROC scores for each classifier
models_info = {}

# Loop over classifiers
for clf_name, clf in classifiers.items():
    print(f"\nClassifier: {clf_name}")
    metrics_list = []
    auroc_list = []
    model_list = []
    feature_names = None  # Initialize feature names
    for split in range(20):
        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(split=split, label_name=label_name, missing_method='median', normalize=True)
        X_train, X_test, y_train, y_test = preprocessor.get_dataset()
        # Ensure the labels are integers
        y_train, y_test = y_train.astype(int), y_test.astype(int)
        
        # Get feature names
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Fit the classifier
        clf.fit(X_train, y_train)
        # Make a copy of the trained model
        model_copy = copy.deepcopy(clf)
        
        # Make predictions on the training set and find the optimal threshold
        if hasattr(clf, "predict_proba"):
            y_prob_train = clf.predict_proba(X_train)[:, 1]
        else:  # For classifiers that do not have predict_proba (e.g., SVM)
            y_prob_train = clf.decision_function(X_train)
        fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Make predictions
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:  # For classifiers that do not have predict_proba (e.g., SVM)
            y_prob = clf.decision_function(X_test)
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob, optimal_threshold)
        metrics_list.append(metrics)
        
        # Store AUROC score
        auroc_list.append(metrics['AUROC'])
        
        # Store model information
        model_info = {
            'split': split,
            'model': model_copy,
            'test_auroc': metrics['AUROC']
        }
        model_list.append(model_info)
        
    # Compute average metrics and standard deviation
    avg_metrics = pd.DataFrame(metrics_list).mean()
    std_metrics = pd.DataFrame(metrics_list).std()
    print("Average Metrics over 20 folds:")
    print(avg_metrics.round(4))
    print("Standard Deviation over 20 folds:")
    print(std_metrics.round(4))
    
    # Store AUROC scores and model information
    auroc_scores[clf_name] = auroc_list
    models_info[clf_name] = model_list

# Perform statistical tests between classifiers
auroc_df = pd.DataFrame(auroc_scores)
from itertools import combinations

classifiers_list = list(classifiers.keys())
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

# For feature importance, pick the model trained on the split that has the highest AUROC on the test data
for clf_name in classifiers:
    if clf_name in ['Random_Forest', 'Logistic_Regression', 'XGBoost']:
        model_list = models_info[clf_name]
        # Convert model_list to DataFrame
        model_df = pd.DataFrame(model_list)
        # Find the index of the model with highest test_auroc
        best_idx = model_df['test_auroc'].idxmax()
        best_model_info = model_list[best_idx]
        best_model = best_model_info['model']
        print(f"\nTop 20 Most Important Features for {clf_name} (Split {best_model_info['split']} with highest AUROC):")
        # Get feature importances
        if clf_name == 'Logistic_Regression':
            importances = np.abs(best_model.coef_[0])
        elif clf_name == 'Random_Forest':
            importances = best_model.feature_importances_
        elif clf_name == 'XGBoost':
            importances = best_model.feature_importances_
        # Create DataFrame of feature importances
        importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importances_df = importances_df.sort_values(by='Importance', ascending=False)
        print(importances_df.head(20).to_string(index=False))
