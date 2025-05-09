import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from .base_classifier import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, name="XGBoost"):
        """
        XGBoost classifier for multi-class problems.
        Trains only on classes present in the training set.
        """
        super().__init__(name=name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.seen_classes = None  # Actual classes seen in training
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_classes = 90

    def train(self, X_train, y_train):
        """
        Trains the XGBoost model using only the classes present in `y_train`.
        The model and label encoder are aligned to this subset.
        """
        # Track and encode only the present classes
        self.seen_classes = np.unique(y_train)
        self.label_encoder.fit(self.seen_classes)
        y_encoded = self.label_encoder.transform(y_train)

        # Initialize and fit the model
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.seen_classes),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        self.model.fit(X_train, y_encoded)

    def classify(self, X):
        """
        Predicts class labels for the input samples.
        Returns the original (unencoded) labels.
        """
        y_encoded_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded_pred)

    def classify_proba(self, X):
        """
        Predicts class probabilities for input samples.
        Returns a (n_samples, 90) matrix.
        Probabilities for classes not seen during training are 0.0.
        """
        y_proba = self.model.predict_proba(X)
        
        # Full set of target classes from 1 to 90
        all_classes = np.arange(1, self.model.n_classes_ + 1)

        # Create encoder for full label set
        full_label_encoder = LabelEncoder()
        full_label_encoder.fit(all_classes)

        # Map trained class indices to full index positions
        seen_class_labels = self.label_encoder.classes_
        full_indices = full_label_encoder.transform(seen_class_labels)

        # Allocate full probability matrix with zeros
        full_proba = np.zeros((X.shape[0], len(all_classes)))

        # Populate the matrix with predicted probabilities
        for i, full_idx in enumerate(full_indices):
            full_proba[:, full_idx] = y_proba[:, i]

        return full_proba

    def save(self, model_dir="models"):
        """
        Saves the model and label encoder to disk.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self._name}.joblib")
        encoder_path = os.path.join(model_dir, f"{self._name}_label_encoder.joblib")

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to: {model_path}")
        print(f"Label encoder saved to: {encoder_path}")

    def load(self, model_dir="models"):
        """
        Loads the model and label encoder from disk.
        """
        model_path = os.path.join(model_dir, f"{self._name}.joblib")
        encoder_path = os.path.join(model_dir, f"{self._name}_label_encoder.joblib")

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"Loaded model from: {model_path}")
            print(f"Loaded label encoder from: {encoder_path}")
        else:
            raise FileNotFoundError("Model or label encoder file not found.")
