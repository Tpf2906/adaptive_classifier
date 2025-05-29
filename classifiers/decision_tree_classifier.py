from sklearn.tree import DecisionTreeClassifier
from .base_classifier import BaseClassifier
from sklearn.calibration import CalibratedClassifierCV

class DecisionTreeClassifierWrapper(BaseClassifier):
    def __init__(self, max_depth=None, random_state=42, name="DecisionTree"):
        """
        Initializes a Decision Tree classifier.

        Parameters:
        - max_depth: The maximum depth of the tree (None means unlimited)
        - random_state: Seed for reproducibility
        """
        super().__init__(name=name)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        self.model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
