import numpy as np
from .BaseClassifier import BaseClassifier


class OwnNearestCentroid(BaseClassifier):

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None):
        super().__init__(X, Y)
        self.centroids = None

    def __calculate_centroids(self):
        self.centroids = np.zeros((self.labels.shape[0], self.X.shape[1]))
        for i in range(self.labels.shape[0]):
            label_associated = np.where(self.Y == self.labels[i])
            class_members = self.X[label_associated]
            self.centroids[i] = np.mean(class_members, axis=0)

    def __calculate_distance(self, entry: np.ndarray, centroid: np.ndarray):
        return np.sqrt(((entry - centroid)**2).sum())

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
