from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base_classifier import BaseClassifier

class LDAClassifier(BaseClassifier):
    def __init__(self, solver='svd', priors=None, shrinkage=None, name="LDA"):
        """
        Initializes the LDA classifier.

        Parameters:
        - solver: Algorithm to use for fitting ('svd', 'lsqr', or 'eigen').
        - priors: Class prior probabilities (used in 'lsqr' and 'eigen' solvers).
        - shrinkage: Regularization strength for 'lsqr' or 'eigen' solvers.
        - name: Identifier for saving/loading the model.
        """
        super().__init__(name=name)
        self.model = LinearDiscriminantAnalysis(
            solver=solver,
            priors=priors,
            shrinkage=shrinkage
        )
