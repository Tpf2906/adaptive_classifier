import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class EnsembleTrainer:
    def __init__(self, controller, classifiers, dataloader, optimizer, alpha=0.0, harmonic_fn=None):
        """
        controller: Neural network model (e.g., WeightController)
        classifiers: list of 11 trained classifier objects with .classify_proba(X)
        dataloader: yields batches of (features_7dim, X_classify_input, labels)
        optimizer: optimizer for controller
        alpha: weighting for the harmonic loss (if used)
        harmonic_fn: callable for additional scoring/loss based on features/probs/labels
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller.to(self.device)
        self.classifiers = classifiers
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.alpha = alpha
        self.harmonic_fn = harmonic_fn

    def train_epoch(self):
        self.controller.train()
        total_loss = 0.0
        for features_7dim, X_input, y_true in self.dataloader:
            features_7dim = features_7dim.to(self.device)
            y_true = y_true.to(self.device)

            # Get classifier predictions (11 x batch_size x 90)
            probas_list = [torch.tensor(clf.classify_proba(X_input), dtype=torch.float32, device=self.device) for clf in self.classifiers]
            probas_stacked = torch.stack(probas_list, dim=1)  # (batch, 11, 90)

            # Get weights from controller
            weights = self.controller(features_7dim)  # (batch, 11)
            weights = F.softmax(weights, dim=1)

            # Apply weights
            weighted_output = torch.sum(weights.unsqueeze(2) * probas_stacked, dim=1)  # (batch, 90)

            # Classification loss
            ce_loss = F.cross_entropy(weighted_output, y_true)

            # Optional harmonic loss
            if self.harmonic_fn is not None:
                harmonic_loss = self.harmonic_fn(features_7dim, weighted_output, y_true)
            else:
                harmonic_loss = 0.0

            total_batch_loss = ce_loss + self.alpha * harmonic_loss

            # Backprop
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        return total_loss / len(self.dataloader)

### Example usage:

# controller = YourWeightController()  # nn.Module
# classifiers = [clf1, clf2, ..., clf11]  # all loaded
# dataloader = DataLoader(...)  # yielding batches of (features, X, y)
# optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)
# trainer = EnsembleTrainer(controller, classifiers, dataloader, optimizer)

for epoch in range(num_epochs):
    loss = trainer.train_epoch()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")