from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import pearsonr
import numpy as np

def compute_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes standard classification metrics and Pearson correlation.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred_probs: Model predicted probabilities (from model.predict)
        threshold: Classification threshold (default=0.5)
        
    Returns:
        dict: metrics, with keys:
            - accuracy
            - precision
            - recall
            - f1
            - auc
            - pearson
            - threshold (used for binary prediction)
    """
    # Ensure inputs are flattened and numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "auc": roc_auc_score(y_true, y_pred_probs),
        "pearson": pearsonr(y_true, y_pred_probs)[0],
        "threshold": threshold
    }
    return metrics
