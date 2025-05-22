# weight_generator.py
from ensembler import EnsemblerClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, roc_auc_score,
    top_k_accuracy_score
)
import numpy as np
from time import time

def get_ensemble_metrics(ensemble: EnsemblerClassifier, X_val, y_val):
    start_time = time.time()
    y_pred = ensemble.classify(X_val)
    elapsed_time = time.time() - start_time
    
    if hasattr(ensemble, "classify_proba"):
        try:
            y_proba = ensemble.classify_proba(X_val)
            top5_acc = top_k_accuracy_score(y_val, y_proba, k=5, labels=np.arange(1, 91))
            # AUC-ROC computation — assumes multi-class OVR
            auc_roc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted', labels=np.arange(1, 91))
        except Exception as e:
            print(f"Probability-based metrics not available for ensemble: {e}")
            top5_acc = None
            auc_roc = None
    else:
        top5_acc = None
        auc_roc = None
    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    precision = precision_score(y_val, y_pred, average='weighted')
    
    return acc, recall, f1, precision, top5_acc, auc_roc, elapsed_time