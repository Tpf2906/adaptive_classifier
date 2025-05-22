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

