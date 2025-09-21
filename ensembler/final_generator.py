# new_weight_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalWeightGeneratorNN(nn.Module):
    def __init__(self, num_metrics=7, num_classifiers=11):
        super(FinalWeightGeneratorNN, self).__init__()
        self.num_classifiers = num_classifiers
        self.input_dim = num_metrics * 7 + 7  # Custom input logic
        self.hidden_dim = 64

        # Layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.activation_layer = nn.Linear(self.hidden_dim, num_classifiers)
        self.weight_layer = nn.Linear(self.hidden_dim, num_classifiers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        activation_logits = self.activation_layer(x)               # shape: (batch_size, num_classifiers)
        weight_logits = self.weight_layer(x)                       # shape: (batch_size, num_classifiers)
        
        activation_mask = torch.sigmoid(activation_logits)        # soft activations between 0 and 1
        weight_scores = F.softmax(weight_logits, dim=1)           # softmax weights summing to 1
        
        # Soft gating: multiply weights by activations
        gated_weights = activation_mask * weight_scores
        
        # Normalize gated weights to sum to 1 to keep it a valid distribution
        gated_weights_sum = gated_weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_weights = gated_weights / gated_weights_sum

        return activation_logits, normalized_weights

    def train_step(self, input_vector, target_weights, optimizer, device=None):
        self.train()
        optimizer.zero_grad()

        if device is not None:
            input_vector = input_vector.to(device)
            target_weights = target_weights.to(device)

        # Forward pass
        activation_logits, predicted_weights = self(input_vector.unsqueeze(0))

        # Use KL divergence loss to better match soft distributions
        loss = F.kl_div(
            (predicted_weights + 1e-8).log(),
            target_weights,
            reduction='batchmean'
        )

        loss.backward()
        optimizer.step()

        return loss.item()

