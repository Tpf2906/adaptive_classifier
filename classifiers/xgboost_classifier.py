import xgboost as xgb
from base_classifier import BaseClassifier

class XGBoostClassifier(BaseClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, name="XGBoost"):
        """
        Initializes the XGBoost classifier.

        Parameters:
        - n_estimators: Number of boosting rounds.
        - learning_rate: Step size shrinking the contribution of each tree.
        - max_depth: Maximum depth of each tree.
        - name: Identifier for saving/loading the model.
        """
        super().__init__(name=name)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
