import numpy as np

# Fungsi untuk menghitung jarak Euclidean
def calculate_euclidean_distance(point1, point2):
    """
    Menghitung jarak Euclidean antara dua titik.
    
    Args:
        point1 (array-like): Titik pertama.
        point2 (array-like): Titik kedua.

    Returns:
        float: Jarak Euclidean.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# Fungsi untuk menemukan tetangga terdekat
def get_nearest_neighbors(features, class_labels, query_point, k_neighbors):
    """
    Menemukan tetangga terdekat berdasarkan jarak Euclidean.
    
    Args:
        features (array-like): Dataset fitur.
        class_labels (array-like): Label kelas untuk dataset.
        query_point (array-like): Data baru untuk prediksi.
        k_neighbors (int): Jumlah tetangga terdekat.

    Returns:
        list: List tetangga terdekat (jarak dan label).
    """
    distances_with_labels = [
        (calculate_euclidean_distance(data_point, query_point), class_labels[index])
        for index, data_point in enumerate(features)
    ]
    distances_with_labels.sort(key=lambda x: x[0])
    return distances_with_labels[:k_neighbors]

# Fungsi untuk melakukan voting manual
def majority_vote(neighbor_labels):
    """
    Melakukan voting untuk menentukan kelas yang paling banyak muncul.
    
    Args:
        neighbor_labels (list): Label kelas dari tetangga terdekat.

    Returns:
        str: Label kelas yang paling banyak muncul.
    """
    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
    majority_index = np.argmax(counts)
    return unique_labels[majority_index]

# Fungsi utama untuk prediksi menggunakan KNN
def knn_predict(features, class_labels, query_point, k_neighbors):
    """
    Prediksi kelas data baru menggunakan algoritma KNN.
    
    Args:
        features (array-like): Dataset fitur.
        class_labels (array-like): Label kelas untuk dataset.
        query_point (array-like): Data baru untuk prediksi.
        k_neighbors (int): Jumlah tetangga terdekat.

    Returns:
        Prediksi kelas dari query_point.
    """
    nearest_neighbors = get_nearest_neighbors(features, class_labels, query_point, k_neighbors)
    neighbor_labels = [label for _, label in nearest_neighbors]
    return majority_vote(neighbor_labels)

# Dataset dengan 10 fitur
feature_data = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
]
class_labels = ['A', 'A', 'A', 'B', 'B', 'B']

# Data yang akan diprediksi
query_point = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

# Prediksi
k_neighbors = 3
predicted_class = knn_predict(feature_data, class_labels, query_point, k_neighbors)
print(f"Data {query_point} diprediksi sebagai kelas '{predicted_class}'.")