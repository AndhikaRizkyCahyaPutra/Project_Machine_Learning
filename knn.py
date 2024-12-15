import math

class KNearestNeighbors:
    def __init__(self, k_neighbors=5):
        """
        Inisialisasi model K-Nearest Neighbors.
        :param k_neighbors: Jumlah tetangga terdekat yang akan dipertimbangkan.
        """
        self.k_neighbors = k_neighbors

    def calculate_distance(self, point_a, point_b):
        """
        Menghitung jarak Euclidean antara dua titik.
        :param point_a: Titik pertama sebagai daftar koordinat.
        :param point_b: Titik kedua sebagai daftar koordinat.
        :return: Jarak Euclidean.
        """
        if len(point_a) != len(point_b):
            raise ValueError("Dimensi kedua titik harus sama.")
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))

    def find_nearest_neighbors(self, feature_data, labels, test_point):
        """
        Temukan tetangga terdekat berdasarkan jarak Euclidean.
        :param feature_data: Data fitur sebagai daftar daftar.
        :param labels: Label kelas untuk setiap titik data.
        :param test_point: Titik yang akan diuji.
        :return: Daftar tetangga terdekat (jarak dan label).
        """
        distances = [
            (self.calculate_distance(data_point, test_point), labels[i])
            for i, data_point in enumerate(feature_data)
        ]
        distances.sort(key=lambda x: x[0])  # Urutkan berdasarkan jarak
        return distances[:self.k_neighbors]

    def determine_majority_class(self, neighbor_labels):
        """
        Menentukan kelas mayoritas dari tetangga terdekat.
        :param neighbor_labels: Daftar label kelas dari tetangga.
        :return: Label kelas dengan jumlah terbanyak.
        """
        label_counts = {}
        for label in neighbor_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)

    def classify(self, feature_data, labels, test_point):
        """
        Klasifikasi titik uji menggunakan K-Nearest Neighbors.
        :param feature_data: Data fitur sebagai daftar daftar.
        :param labels: Label kelas untuk setiap titik data.
        :param test_point: Titik yang akan diuji.
        :return: Prediksi kelas dan informasi tambahan.
        """
        # Temukan tetangga terdekat
        nearest_neighbors = self.find_nearest_neighbors(feature_data, labels, test_point)

        # Ambil label dari tetangga terdekat
        neighbor_labels = [label for _, label in nearest_neighbors]

        # Hitung jumlah setiap kelas
        class_counts = {label: neighbor_labels.count(label) for label in set(labels)}

        # Siapkan data untuk ditampilkan
        neighbors_info = [
            {"distance": round(distance, 4), "class": label}
            for distance, label in nearest_neighbors
        ]

        # Prediksi kelas berdasarkan voting mayoritas
        prediction = self.determine_majority_class(neighbor_labels)

        # Hasil prediksi
        result = {
            "prediction": prediction,
            "nearest_neighbors": neighbors_info,
            "class_counts": class_counts
        }

        # Debugging: Tampilkan informasi tetangga
        print("Tetangga Terdekat:")
        for idx, neighbor in enumerate(neighbors_info, 1):
            print(f"Tetangga {idx}: {neighbor}")

        print("Distribusi Kelas:")
        for cls, count in class_counts.items():
            print(f"Kelas {cls}: {count} tetangga")

        return result
