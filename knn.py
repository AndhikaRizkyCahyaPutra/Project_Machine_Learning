import math

class KNearestNeighbors:
    def __init__(self, k_neighbors=7):
        self.k_neighbors = k_neighbors

    def calculate_distance(self, point_a, point_b):
        if len(point_a) != len(point_b):
            raise ValueError("Dimensi kedua titik harus sama.")
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))

    def find_nearest_neighbors(self, feature_data, labels, test_point):
        distances = [
            (self.calculate_distance(data_point, test_point), labels[i])
            for i, data_point in enumerate(feature_data)
        ]
        distances.sort(key=lambda x: x[0])  
        return distances[:self.k_neighbors]

    def determine_majority_class(self, neighbor_labels):
        label_counts = {}
        for label in neighbor_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)

    def classify(self, feature_data, labels, test_point):
        nearest_neighbors = self.find_nearest_neighbors(feature_data, labels, test_point)

        neighbor_labels = [label for _, label in nearest_neighbors]

        class_counts = {label: neighbor_labels.count(label) for label in set(labels)}

        neighbors_info = [
            {"distance": round(distance, 4), "class": label}
            for distance, label in nearest_neighbors
        ]

        prediction = self.determine_majority_class(neighbor_labels)

        result = {
            "prediction": prediction,
            "nearest_neighbors": neighbors_info,
            "class_counts": class_counts
        }

        print("Tetangga Terdekat:")
        for idx, neighbor in enumerate(neighbors_info, 1):
            print(f"Tetangga {idx}: {neighbor}")

        print("Distribusi Kelas:")
        for cls, count in class_counts.items():
            print(f"Kelas {cls}: {count} tetangga")

        return result
