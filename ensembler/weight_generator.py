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
        activation_logits = self.activation_layer(x)
        weight_scores = F.softmax(self.weight_layer(x), dim=1)
        return activation_logits, weight_scores



def evaluate_ensemble(ensemble: EnsemblerClassifier, X_val, y_val, input_params):
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
    metric_weights = input_params.cpu().numpy().flatten()
    #print(f"Metrics: {metrics}")
    #score = np.dot(metric_weights, metrics)
    
    # Weighted Harmonic mean
    denominator = sum(metric_weights[i] / (metrics[i] + 1e-8) for i in range(7))
    score = metric_weights.sum() / denominator

    return score


def train_step(nn_model: WeightGeneratorNN, optimizer, input_params, X_val, y_val, base_classifiers, device=None, sparsity_coeff=0.01):
    """
    Trains the neural net for one step based on ensemble performance.

    Parameters:
    - input_params: torch.Tensor of shape (1, 7)
    - X_val, y_val: Validation data (NumPy arrays)
    - base_classifiers: list of 11 classifiers
    - sparsity_coeff: float, strength of regularization to reduce number of active classifiers
    """
    nn_model.train()
    optimizer.zero_grad()

    # Move input to device
    input_params = input_params.to(device)

    # Forward pass
    x = F.relu(nn_model.fc1(input_params))
    x = F.relu(nn_model.fc2(x))

    activation_logits = nn_model.activation_layer(x)
    weight_logits = nn_model.weight_layer(x)

    activation_probs = torch.sigmoid(activation_logits)
    weight_scores = F.softmax(weight_logits, dim=1)

    # Sample binary activations
    bernoulli_dist = torch.distributions.Bernoulli(probs=activation_probs)
    sampled_activation = bernoulli_dist.sample()
    log_probs = bernoulli_dist.log_prob(sampled_activation)

    # Ensure at least one classifier is active
    if sampled_activation.sum().item() < 1:
        max_idx = torch.argmax(activation_probs, dim=1)
        # Clone to avoid in-place ops
        sampled_activation = sampled_activation.clone()
        log_probs = log_probs.clone()
        
        # Use index_fill for safety
        sampled_activation[0].index_fill_(0, max_idx, 1.0)
        log_probs[0].index_fill_(0, max_idx, torch.log(activation_probs[0, max_idx] + 1e-8))

    # Convert activation & weights to NumPy
    activation_np = sampled_activation.detach().cpu().numpy().flatten()
    weights_np = weight_scores.detach().cpu().numpy().flatten()

    # Filter selected classifiers
    selected = [(clf, weight) for clf, act, weight in zip(base_classifiers, activation_np, weights_np) if act >= 0.5]

    # Normalize weights
    total_weight = sum(w for _, w in selected)
    if total_weight > 0:
        selected = [(clf, w / total_weight) for clf, w in selected]

    # Build and evaluate ensemble
    ensemble = EnsemblerClassifier(selected)
    reward = evaluate_ensemble(ensemble, X_val, y_val, input_params)

    # Loss = -reward * log_probs + sparsity penalty
    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
    activation_count = sampled_activation.sum()
    sparsity_penalty = sparsity_coeff * activation_count
    loss = -reward_tensor * log_probs.sum() + sparsity_penalty

    # Backpropagation
    loss.backward()
    optimizer.step()

    return reward, loss.item()

