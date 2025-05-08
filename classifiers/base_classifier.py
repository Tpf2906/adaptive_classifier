import joblib
import os
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def __init__(self, name="base"):
        """
        Initializes the classifier with an optional name.

        Parameters:
        - name: Optional name for the classifier
        """
        self._name = name
        self.model = None # Subclass should initialize the model
        
    def train(self, X_train, y_train):
        """
        Train the SVM model.

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        """
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model.")
        self.model.fit(X_train, y_train)

    def classify(self, X):
        """
        Classify input features using the trained SVM.

        Parameters:
        - X: Input features

        Returns:
        - Predicted class labels
        """
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model.")
        return self.model.predict(X)
    
    def save(self, model_dir="models"):
        """
        Save the trained model to disk using the name attribute.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filepath = os.path.join(model_dir, f"{self.name}.joblib")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, model_dir="models"):
        """
        Load a trained model from disk using the name attribute.
        """
        filepath = os.path.join(model_dir, f"{self.name}.joblib")
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found!")
