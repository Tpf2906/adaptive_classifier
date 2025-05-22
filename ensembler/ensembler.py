import numpy as np
from typing import List, Tuple
from classifiers import BaseClassifier

class EnsemblerClassifier:
    def __init__(self, classifiers_with_weights: List[Tuple[BaseClassifier, float]]):
        """
        Ensemble classifier using weighted average of predicted probabilities.

        Parameters:
        - classifiers_with_weights: List of (classifier_instance, weight)
        """
        self.classifiers = [clf for clf, _ in classifiers_with_weights]
        self.weights = np.array([w for _, w in classifiers_with_weights], dtype=np.float32)

        if len(self.classifiers) == 0:
            raise ValueError("No classifiers provided.")
        if len(self.classifiers) != len(self.weights):
            raise ValueError("Mismatch between classifiers and weights.")

    def classify_proba(self, X):
        """
        Combine probability predictions from classifiers using weighted average.

        Returns:
        - Combined (n_samples, n_classes) probability matrix
        """
        weighted_probs = None

        for clf, weight in zip(self.classifiers, self.weights):
            probs = clf.classify_proba(X)
            if weighted_probs is None:
                weighted_probs = weight * probs
            else:
                weighted_probs += weight * probs

        return weighted_probs / self.weights.sum()

    def classify(self, X):
        """
        Predict final class using highest combined probability.
        """
        combined_proba = self.classify_proba(X)
        return np.argmax(combined_proba, axis=1)

