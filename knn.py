import numpy as np
class KNN:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors

    # Fungsi untuk menghitung jarak Euclidean
    def calculate_euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

    # Fungsi untuk menemukan tetangga terdekat
    def get_nearest_neighbors(self, features, class_labels, query_point):
        distances_with_labels = [
            (self.calculate_euclidean_distance(data_point, query_point), class_labels[index])
            for index, data_point in enumerate(features)
        ]
        distances_with_labels.sort(key=lambda x: x[0])  # Urutkan berdasarkan jarak terdekat
        return distances_with_labels[:self.k_neighbors]

    # Fungsi untuk melakukan voting mayoritas
    def majority_vote(self, neighbor_labels):
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        majority_index = np.argmax(counts)
        return unique_labels[majority_index]

    # Fungsi utama untuk prediksi menggunakan KNN
    def predict(self, features, class_labels, query_point):
        nearest_neighbors = self.get_nearest_neighbors(features, class_labels, query_point)

        # Menampilkan top-k tetangga terdekat dan jaraknya
        print("Top-k Tetangga Terdekat dan Jaraknya:")
        for i, (distance, label) in enumerate(nearest_neighbors):
            print(f"Tetangga {i+1}: Kelas={label}, Jarak={distance:.4f}")

        # Menghitung jumlah kelas 0 dan kelas 1
        neighbor_labels = [label for _, label in nearest_neighbors]
        class_0_count = neighbor_labels.count(0)  # Hitung kelas 0
        class_1_count = neighbor_labels.count(1)  # Hitung kelas 1

        # Menampilkan jumlah kelas 0 dan kelas 1 di antara top-k tetangga terdekat
        print(f"Jumlah kelas 0 di top-{self.k_neighbors} tetangga terdekat: {class_0_count}")
        print(f"Jumlah kelas 1 di top-{self.k_neighbors} tetangga terdekat: {class_1_count}")

        # Prediksi berdasarkan mayoritas kelas dari top-k tetangga
        return self.majority_vote(neighbor_labels)

    # Fungsi untuk menghitung metrik evaluasi
    def calculate_metrics(self, true_labels, predicted_labels):
        """
        Menghitung presisi, recall, dan akurasi.

        Args:
            true_labels (array-like): Label sebenarnya.
            predicted_labels (array-like): Label prediksi.

        Returns:
            dict: Dictionary dengan presisi, akurasi, dan recall.
        """
        unique_labels = np.unique(true_labels)
        metrics = {}
        for label in unique_labels:
            true_positive = sum((np.array(predicted_labels) == label) & (np.array(true_labels) == label))
            false_positive = sum((np.array(predicted_labels) == label) & (np.array(true_labels) != label))
            false_negative = sum((np.array(predicted_labels) != label) & (np.array(true_labels) == label))
            true_negative = len(true_labels) - (true_positive + false_positive + false_negative)

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            accuracy = (true_positive + true_negative) / len(true_labels)

            metrics[label] = {
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
            }

        return metrics