import numpy as np
import torch
import random
import os

# Function to set the random seed for reproducibility
def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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