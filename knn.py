import numpy as np

class KNN:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors

    def calculate_euclidean_distance(self, point1, point2):
        """Menghitung jarak Euclidean antara dua titik."""
        point1 = np.array(point1, dtype=float)
        point2 = np.array(point2, dtype=float)
        return np.sqrt(np.sum((point1 - point2)**2))

    def get_nearest_neighbors(self, features, class_labels, query_point):
        """Menemukan tetangga terdekat berdasarkan jarak."""
        distances_with_labels = [
            (self.calculate_euclidean_distance(data_point, query_point), class_labels[index])
            for index, data_point in enumerate(features)
        ]
        distances_with_labels.sort(key=lambda x: x[0])
        return distances_with_labels[:self.k_neighbors]

    def majority_vote(self, neighbor_labels):
        """Melakukan voting mayoritas untuk menentukan prediksi."""
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        majority_index = np.argmax(counts)
        return unique_labels[majority_index]

    def predict(self, features, class_labels, query_point):
        """Prediksi kelas untuk sebuah titik uji."""
        # Temukan top-k tetangga terdekat
        nearest_neighbors = self.get_nearest_neighbors(features, class_labels, query_point)

        # Siapkan data untuk ditampilkan
        neighbors_data = [{"class": label, "distance": f"{distance:.4f}"} for distance, label in nearest_neighbors]

        # Hitung jumlah masing-masing kelas di antara top-k tetangga
        neighbor_labels = [label for _, label in nearest_neighbors]
        class_counts = {label: neighbor_labels.count(label) for label in set(class_labels)}

        # Kembalikan hasil prediksi serta informasi tetangga terdekat dan jumlah kelas
        result = {
            "prediction": self.majority_vote(neighbor_labels),
            "nearest_neighbors": neighbors_data,
            "class_counts": class_counts
        }

        # Debugging: Tampilkan tetangga terdekat dan jaraknya
        print("Top-k Tetangga Terdekat dan Jaraknya:")
        for i, (distance, label) in enumerate(nearest_neighbors):
            print(f"Tetangga {i+1}: Kelas={label}, Jarak={distance:.4f}")

        # Debugging: Tampilkan jumlah masing-masing kelas
        print("Jumlah masing-masing kelas di top-k tetangga:")
        for cls, count in class_counts.items():
            print(f"Kelas {cls}: {count} tetangga")

        return result
