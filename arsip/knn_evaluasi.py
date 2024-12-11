import numpy as np

# Fungsi untuk menghitung jarak Euclidean
def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# Fungsi untuk menemukan tetangga terdekat
def get_nearest_neighbors(features, class_labels, query_point, k_neighbors):
    distances_with_labels = [
        (calculate_euclidean_distance(data_point, query_point), class_labels[index])
        for index, data_point in enumerate(features)
    ]
    distances_with_labels.sort(key=lambda x: x[0])
    return distances_with_labels[:k_neighbors]

# Fungsi untuk melakukan voting manual
def majority_vote(neighbor_labels):
    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
    majority_index = np.argmax(counts)
    return unique_labels[majority_index]

# Fungsi utama untuk prediksi menggunakan KNN
def knn_predict(features, class_labels, query_point, k_neighbors):
    nearest_neighbors = get_nearest_neighbors(features, class_labels, query_point, k_neighbors)
    neighbor_labels = [label for _, label in nearest_neighbors]
    return majority_vote(neighbor_labels)

# Membuat dataset dengan 45 data (10 fitur dan 2 kelas)
np.random.seed(42)
feature_data = np.random.randint(1, 100, size=(45, 10))
class_labels = np.random.choice(['A', 'B'], size=45)

# Membagi dataset menjadi train (80%) dan test (20%)
train_size = int(0.8 * len(feature_data))
test_size = len(feature_data) - train_size

train_features = feature_data[:train_size]
train_labels = class_labels[:train_size]
test_features = feature_data[train_size:]
test_labels = class_labels[train_size:]

# KNN pada data uji
k_neighbors = 5
predictions = [knn_predict(train_features, train_labels, test_point, k_neighbors) for test_point in test_features]

# Menghitung metrik evaluasi
def calculate_metrics(true_labels, predicted_labels):
    """
    Menghitung presisi, akurasi, dan recall.

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

metrics = calculate_metrics(test_labels, predictions)

# Menampilkan hasil
print("Hasil Prediksi:", predictions)
print("Label Sebenarnya:", test_labels.tolist())
print("\nMetrik Evaluasi:")
for label, label_metrics in metrics.items():
    print(f"Kelas '{label}':")
    for metric_name, value in label_metrics.items():
        print(f"  {metric_name}: {value:.2f}")
