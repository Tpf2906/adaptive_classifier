# weight_generator.py
from .ensembler import EnsemblerClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, roc_auc_score,
    top_k_accuracy_score
)
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import time
import warnings

def get_ensemble_metrics(ensemble: EnsemblerClassifier, X_val, y_val):
    start_time = time.time()
    y_pred = ensemble.classify(X_val)
    elapsed_time = time.time() - start_time

    top5_acc = None
    auc_roc = None

    if hasattr(ensemble, "classify_proba"):
        try:
            y_proba = ensemble.classify_proba(X_val)
            top5_acc = top_k_accuracy_score(y_val, y_proba, k=5, labels=np.arange(1, 91))

            # Only compute AUC if y_val has at least 2 unique classes
            if len(np.unique(y_val)) > 1:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    auc_roc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted', labels=np.arange(1, 91))
        except Exception as e:
            top5_acc = None
            auc_roc = None

    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)

    return acc, recall, f1, precision, top5_acc, auc_roc, elapsed_time
