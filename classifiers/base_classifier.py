import joblib
import os
from abc import ABC
import numpy as np
from sklearn.preprocessing import LabelEncoder

class BaseClassifier(ABC):
    def __init__(self, name="base", n_classes=90):
        """
        Initializes the classifier with an optional name.

        Parameters:
        - name: Optional name for the classifier
        """
        self._name = name
        self.model = None # Subclass should initialize the model
        self.label_encoder = LabelEncoder()
        self.seen_classes = None
        self.n_classes = n_classes
        
    def train(self, X_train, y_train):
        """
        Train the SVM model.

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        """
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model.")
        self.seen_classes = np.unique(y_train)
        self.label_encoder.fit(self.seen_classes)
        y_encoded = self.label_encoder.transform(y_train)
        self.model.fit(X_train, y_encoded)

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
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def classify_proba(self, X):
        """
        Predict class probabilities for input features.

        Parameters:
        - X: Input features

        Returns:
        - Predicted class probabilities
        """
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model.")
        y_proba = self.model.predict_proba(X)

        # Full class range (assuming classes are 1-based: 1 to 90)
        all_classes = np.arange(1, self.n_classes + 1)

        # Encoder for full label space
        full_label_encoder = LabelEncoder()
        full_label_encoder.fit(all_classes)

        # Map trained (seen) class indices to positions in full matrix
        seen_class_labels = self.label_encoder.classes_
        full_indices = full_label_encoder.transform(seen_class_labels)

        full_proba = np.zeros((X.shape[0], len(all_classes)))
        for i, full_idx in enumerate(full_indices):
            full_proba[:, full_idx] = y_proba[:, i]

        return full_proba
    
    def save(self, model_dir="models_pca"):
        """
        Save the trained model to disk using the name attribute.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self._name}.joblib")
        encoder_path = os.path.join(model_dir, "label_encoder.joblib")

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to: {model_path}")
        print(f"Label encoder saved to: {encoder_path}")

    def load(self, model_dir="models_pca"):
        """
        Load a trained model from disk using the name attribute.
        """
        model_path = os.path.join(model_dir, f"{self._name}.joblib")
        encoder_path = os.path.join(model_dir, "label_encoder.joblib")

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"Loaded model from: {model_path}")
            print(f"Loaded label encoder from: {encoder_path}")
        else:
            raise FileNotFoundError("Model or label encoder file not found.")
