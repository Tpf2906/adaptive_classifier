# weight_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ensembler import EnsemblerClassifier
import numpy as np
from sklearn.metrics import make_scorer
from time import time
from .utils import get_ensemble_metrics

class WeightGeneratorNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, num_classifiers=11):
        super(WeightGeneratorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.activation_layer = nn.Linear(hidden_dim, num_classifiers)
        self.weight_layer = nn.Linear(hidden_dim, num_classifiers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Use straight-through estimator for binary activation
        logits = self.activation_layer(x)
        probs = torch.sigmoid(logits)
        hard = (probs > 0.5).float()
        activation = (hard - probs).detach() + probs  # straight-through

        weights = F.softmax(self.weight_layer(x), dim=1)

        return activation, weights


def evaluate_ensemble(ensemble: EnsemblerClassifier, X_val, y_val, weights):
    """
    Evaluates the ensemble based on user preferences (input_params).

    Parameters:
    - ensemble: A classifier with weighted base classifiers.
    - input_params: 1x7 torch tensor (user-defined importance of each metric)

    Returns:
    - score: scalar reward value for use in training
    """
    # Get metrics
    acc, recall, f1, precision, top5_acc, auc_roc, elapsed_time = get_ensemble_metrics(
        ensemble, X_val, y_val
    )

    # Normalize elapsed time (lower is better → higher score)
    time_score = 1 / (1 + elapsed_time)

    # Final score as weighted arithmetic mean
    metrics = [acc, recall, f1, precision, top5_acc, auc_roc, time_score]
    print(f"Metrics: {metrics}")
    score = np.dot(weights, metrics)
    
    # Weighted Harmonic mean
    # denominator = sum(metric_weights[i] / (metrics[i] + 1e-8) for i in range(7))
    # score = metric_weights.sum() / denominator

    return score


def train_step(nn_model: WeightGeneratorNN, optimizer, input_params, X_val, y_val, base_classifiers, device=None):
    """
    Trains the neural net for one step based on ensemble performance.

    Parameters:
    - input_params: torch.Tensor of shape (1, 7)
    - X_val, y_val: Validation data (NumPy arrays)
    - base_classifiers: list of 11 classifiers
    """
    nn_model.train()
    optimizer.zero_grad()

    # Forward pass
    activation, weights = nn_model(input_params.to(device))  # Both are shape (1, 11)

    # Ensure at least one classifier is active
    if activation.sum().item() < 1:
        # Force the classifier with the highest activation probability to be active
        max_idx = torch.argmax(activation)
        activation[0][max_idx] = 1.0

    # Detach and convert to NumPy
    activation_np = activation.detach().cpu().numpy().flatten()
    weights_np = weights.detach().cpu().numpy().flatten()

    # Filter active classifiers and their weights
    selected = [(clf, weight) for clf, act, weight in zip(base_classifiers, activation_np, weights_np) if act >= 0.5]

    # Normalize selected weights
    total_weight = sum(w for _, w in selected)
    if total_weight > 0:
        selected = [(clf, w / total_weight) for clf, w in selected]

    # Build and evaluate ensemble
    ensemble = EnsemblerClassifier(selected)
    score = evaluate_ensemble(ensemble, X_val, y_val, np.array([w for _, w in selected]))

    # Convert score to loss (maximize score → minimize -score)
    loss = -torch.tensor(score, dtype=torch.float32, device=device, requires_grad=True)

    loss.backward()
    optimizer.step()

    return score, loss.item()
