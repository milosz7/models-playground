import numpy as np
from algorithms.BaseClassifier import BaseClassifier
from typing import Literal


class OwnNearestNeighbors(BaseClassifier):

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None):
        super().__init__(X, Y)
        self.best_distances = None
        self.best_points = None
        self.labels = None

    def __distance(self, point1, point2):
        return ((point1 - point2)**2).sum()

    def __closer_distance(self, new_point, point1, point2):
        if point2 is None:
            return point1
        if point1 is None:
            return point2
        d1 = self.__distance(new_point, point1)
        d2 = self.__distance(new_point, point2)
        if (d1 < d2):
            return point1
        else:
            return point2

    def __is_unique(self, point):
        return (np.where(np.all(self.best_points == point,
                                axis=1))[0].size == 0)

    def __update_best(self, point, best):
        worst_distance_idx = np.argmax(self.best_distances)
        point_distance = self.__distance(point, best)
        if (point_distance < self.__distance(
                point, self.best_distances[worst_distance_idx])):
            self.best_distances[worst_distance_idx] = point_distance
            self.best_points[worst_distance_idx] = best

    def __kd_tree(self, X: np.ndarray, Y: np.ndarray, depth: int = 0):
        if not X.any():
            return None
        else:
            n_features = X.shape[1]
            axis = depth % n_features
            sorted_ids = X[:, axis].argsort()
            X = X[sorted_ids]
            Y = Y[sorted_ids]
            split_node = X[X.shape[0] // 2, :]
            split_label = Y[Y.shape[0] // 2]
            node = {"axis": axis, "location": split_node, "label": split_label}
            node["left"] = self.__kd_tree(X[:(X.shape[0] // 2), :],
                                          Y[:(Y.shape[0] // 2)], depth + 1)
            node["right"] = self.__kd_tree(X[(X.shape[0] // 2 + 1):, :],
                                           Y[(Y.shape[0] // 2 + 1):],
                                           depth + 1)
            return node

    def __vote(self, labels):
        return np.argmax(np.bincount(labels))

    def __bruteforce(self, point, point_idx, n_neighbors):
        best_distances = np.full(n_neighbors, np.inf)
        best_labels = np.zeros(n_neighbors, dtype="uint8")
        for i in range(self.X.shape[0]):
            distance = self.__distance(self.X[i], point)
            max_distance_idx = np.argmax(best_distances)
            if distance < best_distances[np.argmax(best_distances)]:
                best_distances[np.argmax(best_distances)] = distance
                best_labels[max_distance_idx] = self.Y[i]
        self.labels[point_idx] = self.__vote(best_labels)

    def __traverse_kdtree(self, current, point, depth=0):
        if current is None:
            return None

        n_features = point.shape[0]
        axis = depth % n_features
        next_node = None
        opposite_node = None

        if (point[axis] < current["location"][axis]):
            next_node = current["left"]
            opposite_node = current["right"]
        else:
            next_node = current["right"]
            opposite_node = current["left"]

        best = self.__closer_distance(
            point, self.__traverse_kdtree(next_node, point, depth + 1),
            current["location"])

        if (self.__is_unique(best)):
            self.__update_best(point, best)

        worst_distance = self.best_distances[np.argmax(self.best_distances)]

        if (self.__distance(point, worst_distance) >
                abs(point[axis] - current["location"][axis])):
            best = self.__closer_distance(
                point, self.__traverse_kdtree(opposite_node, point, depth + 1),
                best)

        if (self.__is_unique(best)):
            self.__update_best(point, best)

        return best

    def predict(self,
                X_test: np.ndarray,
                n_neigbors: int = 1,
                algorithm: Literal["kdtree", "bruteforce"] = "kdtree"):
        self.labels = np.zeros(X_test.shape[0], dtype="uint8")
        if (algorithm == "kdtree"):
            tree = self.__kd_tree(self.X, self.Y)
            for i in range(X_test.shape[0]):
                self.best_points = np.zeros((n_neigbors, 4))
                self.best_distances = np.full(n_neigbors, np.inf)
                self.__traverse_kdtree(tree, X_test[i], i)
                voting_labels = np.zeros(n_neigbors, dtype="uint8")
                for j in range(n_neigbors):
                    voting_labels[j] = self.Y[np.where(
                        np.all(self.X_test == self.best_points[j], axis=1))][0]
                self.labels[i] = self.__vote(voting_labels)
        if (algorithm == "bruteforce"):
            for i in range(X_test.shape[0]):
                self.__bruteforce(X_test[i], i, n_neigbors)
        return self.labels
