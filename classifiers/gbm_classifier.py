from sklearn.ensemble import GradientBoostingClassifier
from base_classifier import BaseClassifier

class GBMClassifier(BaseClassifier):
    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=5, name="GBM"):
        """
        Initializes the Gradient Boosting Classifier.

        Parameters:
        - n_estimators: Number of boosting stages.
        - learning_rate: Learning rate shrinks the contribution of each tree.
        - max_depth: Maximum depth of individual estimators.
        - name: Identifier name for saving/loading the model.
        """
        super().__init__(name=name)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
