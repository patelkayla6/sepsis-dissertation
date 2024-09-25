import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

# Ignore warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
set_seed(42)

# Define the label name (e.g., 'DIED', 'HFNOV', 'MEVEN')
label_name = 'MEVEN'
print(f"Label: {label_name}")

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
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
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'F1-score': f1,
        'AUROC': auroc,
        'AUPRC': auprc
    }


# Loop over classifiers
for clf_name, clf in classifiers.items():
    print(f"\nClassifier: {clf_name}")
    metrics_list = []
    for split in range(20):
        # Initialize the data preprocessor
        preprocessor = DataPreprocessor(split=split, label_name=label_name, missing_method='median', normalize=True)
        X_train, X_test, y_train, y_test = preprocessor.get_dataset()
        # Ensure the labels are integers
        y_train, y_test = y_train.astype(int), y_test.astype(int)

        # Fit the classifier
        clf.fit(X_train, y_train)

        # Make predictions on the training set and find the optimal threshold
        if hasattr(clf, "predict_proba"):
            y_prob_train = clf.predict_proba(X_train)[:, 1]
        else:  # For classifiers that do not have predict_proba (e.g., SVM)
            y_prob_train = clf.decision_function(X_train)
        fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Make predictions
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:  # For classifiers that do not have predict_proba (e.g., SVM)
            y_prob = clf.decision_function(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob, optimal_threshold)
        metrics_list.append(metrics)
    
    # Compute average metrics and standard deviation
    avg_metrics = pd.DataFrame(metrics_list).mean()
    std_metrics = pd.DataFrame(metrics_list).std()
    print("Average Metrics over 20 folds:")
    print(avg_metrics.round(4))
    print("Standard Deviation over 20 folds:")
    print(std_metrics.round(4))
    
    # # Train classifier on the whole dataset
    # clf.fit(X, y)
    
    # # Feature importance (if available)
    # if clf_name in ['Random Forest', 'Logistic Regression', 'XGBoost']:
    #     if clf_name == 'Logistic Regression':
    #         importances = np.abs(clf.coef_[0])
    #     else:
    #         importances = clf.feature_importances_
        
    #     feature_names = X.columns
    #     importances_df = pd.DataFrame({
    #         'Feature': feature_names,
    #         'Importance': importances
    #     })
    #     importances_df = importances_df.sort_values(by='Importance', ascending=False)
        
    #     print("\nTop 20 Most Important Features:")
    #     print(importances_df.head(20).to_string(index=False))
