# weight_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ensembler import EnsemblerClassifier
import numpy as np
from sklearn.metrics import make_scorer
from time import time
from utils import get_ensemble_metrics

class WeightGeneratorNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=11, hidden_dim=32):
        super(WeightGeneratorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output_layer(x), dim=1)


def evaluate_ensemble(ensemble: EnsemblerClassifier, X_val, y_val, input_params):
    """
    Evaluates the ensemble based on user preferences (input_params).

    Parameters:
    - ensemble: A classifier with weighted base classifiers.
    - input_params: 1x7 torch tensor (user-defined importance of each metric)

    Returns:
    - score: scalar reward value for use in training
    """
    # Softmax to normalize user preferences
    metric_weights = F.softmax(input_params, dim=1).detach().cpu().numpy().flatten()

    # Get metrics
    acc, recall, f1, precision, top5_acc, auc_roc, elapsed_time = get_ensemble_metrics(
        ensemble, X_val, y_val
    )

    # Normalize elapsed time (lower is better → higher score)
    time_score = 1 / (1 + elapsed_time)

    # Final score as weighted arithmetic mean
    metrics = [acc, recall, f1, precision, top5_acc, auc_roc, time_score]
    score = np.dot(metric_weights, metrics)
    
    # Weighted Harmonic mean
    # denominator = sum(metric_weights[i] / (metrics[i] + 1e-8) for i in range(7))
    # score = metric_weights.sum() / denominator

    return score


def train_step(nn_model: WeightGeneratorNN, optimizer, input_params, X_val, y_val, base_classifiers):
    """
    Trains the neural net for one step based on ensemble performance.

    Parameters:
    - input_params: torch.Tensor of shape (1, 7)
    - X_val, y_val: Validation data (NumPy arrays)
    - base_classifiers: list of 11 classifiers
    """
    nn_model.train()
    optimizer.zero_grad()

    # Generate weights from NN
    weights = nn_model(input_params)
    weights_np = weights.detach().cpu().numpy().flatten()

    # Create ensemble with generated weights
    clf_with_weights = list(zip(base_classifiers, weights_np))
    ensemble = EnsemblerClassifier(clf_with_weights)

    # Evaluate performance
    score = evaluate_ensemble(ensemble, X_val, y_val, input_params)

    # Convert score to loss (maximize score → minimize -score)
    loss = -torch.tensor(score, requires_grad=True)

    loss.backward()
    optimizer.step()

    return score, loss.item()
