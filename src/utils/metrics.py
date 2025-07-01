"""Metric helpers for binary phishing detection."""
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_fscore_support

def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
    }