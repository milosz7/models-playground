import numpy as np


class OwnNearestCentroid:

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None):
        self.X = X
        self.Y = Y
        self.centroids = None
        self.estimations = None
        if (Y is not None):
            self.labels = np.sort(np.unique(Y))
        else:
            self.labels = None

    def __calculate_centroids(self):
        self.centroids = np.zeros((self.labels.shape[0], self.X.shape[1]))
        for i in range(self.labels.shape[0]):
            label_associated = np.where(self.Y == self.labels[i])
            class_members = self.X[label_associated]
            self.centroids[i] = np.mean(class_members, axis=0)

    def __calculate_distance(self, entry: np.ndarray, centroid: np.ndarray):
        return np.sqrt(((entry - centroid)**2).sum())

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if (not X.any() or not Y.any()):
            raise ValueError("Argument arrays cannot be empty.")
        if (X.shape[0] == Y.shape[0]):
            self.X = X
            self.Y = Y
            self.labels = np.sort(np.unique(Y))
        else:
            raise ValueError("Data and labels array size are different.")

    def predict(self, X: np.ndarray):
        if (self.X is None or self.Y is None):
            raise TypeError("Provide data using fit() method.")
        
        self.__calculate_centroids()
        
        if (X.shape[1] != self.centroids.shape[1]):
            raise ValueError("Input data and centroid shapes do not match.")
        
        self.estimations = np.zeros(X.shape[0], dtype="uint8")

        for i in range(X.shape[0]):
            distances = [
                self.__calculate_distance(X[i], centroid)
                for centroid in self.centroids
            ]
            label = self.labels[np.argmin(distances)]
            self.estimations[i] = label
        
        return self.estimations

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
