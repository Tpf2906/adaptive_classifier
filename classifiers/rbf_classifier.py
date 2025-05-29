from sklearn.svm import SVC
from .base_classifier import BaseClassifier
from sklearn.calibration import CalibratedClassifierCV

class RBFClassifier(BaseClassifier):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', name="RBF"):
        """
        Initializes the SVM classifier.

        Parameters:
        - kernel: Kernel type (e.g., 'linear', 'rbf')
        - C: Regularization parameter
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        """
        super().__init__(name=name)
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
