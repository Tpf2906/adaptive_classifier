from sklearn.naive_bayes import GaussianNB
from .base_classifier import BaseClassifier

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, name="NaiveBayes"):
        """
        Initializes the Naive Bayes classifier.
        """
        super().__init__(name=name)
        self.model = GaussianNB()
