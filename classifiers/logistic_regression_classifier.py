from sklearn.linear_model import LogisticRegression
from base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, C=1.0, max_iter=1000, name="LogisticRegression"):
        """
        Initializes the Logistic Regression classifier.

        Parameters:
        - C: Inverse of regularization strength
        - max_iter: Maximum number of iterations for optimization
        - name: Name for saving/loading the model
        """
        super().__init__(name=name)
        self.model = LogisticRegression(C=C, max_iter=max_iter)
