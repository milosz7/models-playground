import numpy as np


class BaseClassifier():

    def __init__(self, X: np.ndarray | None, Y: np.ndarray | None):
        self.X = X
        self.Y = Y
        self.estimations = None
        if (Y is not None):
            self.labels = np.sort(np.unique(Y))
        else:
            self.labels = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if (not X.any() or not Y.any()):
            raise ValueError("Argument arrays cannot be empty.")
        if (X.shape[0] == Y.shape[0]):
            self.X = X
            self.Y = Y
            self.labels = np.sort(np.unique(Y))
        else:
            raise ValueError("Data and labels array size are different.")

    def score(self, Y: np.ndarray):
        if (self.estimations is None):
            TypeError(
                "To calculate the score you have to use the predict() method first."
            )
        if (self.estimations.shape[0] != Y.shape[0]):
            ValueError(
                "Provided labels amount does not match estimated labels amount."
            )

        correct_estimations = 0
        entries_amount = Y.shape[0]
        for (label, estimation) in zip(Y, self.estimations):
            if (label == estimation):
                correct_estimations += 1
        return correct_estimations / entries_amount
