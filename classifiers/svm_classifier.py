from sklearn.svm import SVC
from .base_classifier import BaseClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import joblib

class SVMClassifier(BaseClassifier):
    def __init__(self, kernel='linear', C=1.0, gamma='scale', name="SVM"):
        """
        Initializes the SVM classifier.

        Parameters:
        - kernel: Kernel type (e.g., 'linear', 'rbf')
        - C: Regularization parameter
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        """
        super().__init__(name=name)
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        