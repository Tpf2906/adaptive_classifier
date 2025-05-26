from sklearn.ensemble import AdaBoostClassifier
from .base_classifier import BaseClassifier
from sklearn.calibration import CalibratedClassifierCV

class AdaBoostClassifierWrapper(BaseClassifier):
    def __init__(self, n_estimators=75, learning_rate=1.0, name="AdaBoost"):
        """
        Initializes the AdaBoost classifier.

        Parameters:
        - n_estimators: The maximum number of estimators at which boosting is terminated.
        - learning_rate: Weight applied to each classifier at each boosting iteration.
        - name: Identifier for saving/loading the model.
        """
        super().__init__(name=name)
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        self.model = CalibratedClassifierCV(model, method='isotonic', cv=3)
