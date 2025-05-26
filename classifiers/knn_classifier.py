from sklearn.neighbors import KNeighborsClassifier
from .base_classifier import BaseClassifier
from sklearn.calibration import CalibratedClassifierCV

class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors=5, name="KNN"):
        """
        Initializes the k-NN classifier.

        Parameters:
        - n_neighbors: Number of neighbors to use
        - name: Name for saving/loading the model
        """
        super().__init__(name=name)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model = CalibratedClassifierCV(model, method='isotonic', cv=3)
