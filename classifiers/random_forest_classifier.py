from sklearn.ensemble import RandomForestClassifier as RFC
from .base_classifier import BaseClassifier

class RandomForestClassifierWrapper(BaseClassifier):
    def __init__(self, n_estimators=200, max_depth=10, random_state=42, name="RandomForest"):
        """
        Initializes a Random Forest classifier.

        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - random_state: For reproducibility
        """
        super().__init__(name=name)
        self.model = RFC(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

