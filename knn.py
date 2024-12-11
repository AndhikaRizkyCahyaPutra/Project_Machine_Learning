import numpy as np

class KNN:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors

    @staticmethod
    def calculate_euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

    def get_nearest_neighbors(self, features, class_labels, query_point):
        distances_with_labels = [
            (self.calculate_euclidean_distance(data_point, query_point), class_labels[index])
            for index, data_point in enumerate(features)
        ]
        distances_with_labels.sort(key=lambda x: x[0])
        return distances_with_labels[:self.k_neighbors]

    @staticmethod
    def majority_vote(neighbor_labels):
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        majority_index = np.argmax(counts)
        return unique_labels[majority_index]

    def predict(self, features, class_labels, query_point):
        nearest_neighbors = self.get_nearest_neighbors(features, class_labels, query_point)
        neighbor_labels = [label for _, label in nearest_neighbors]
        return self.majority_vote(neighbor_labels)
