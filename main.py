from data_preprocessing import DataPreprocessing
from knn import KNN
from evaluation import Evaluation
import numpy as np

# Membaca dan memproses data
file_path = 'enhanced_fever_medicine_recommendation.csv'
preprocessor = DataPreprocessing(file_path)
data_after_preprocessing = preprocessor.preprocess()

# Memisahkan fitur dan label
class_labels = data_after_preprocessing.iloc[:, -1].values
features = data_after_preprocessing.iloc[:, :-1].values

# Membagi dataset menjadi train (80%) dan test (20%)
train_size = int(0.8 * len(features))
train_features = features[:train_size]
train_labels = class_labels[:train_size]
test_features = features[train_size:]
test_labels = class_labels[train_size:]

# Membuat instance KNN
k_neighbors = 5
knn = KNN(k_neighbors=k_neighbors)

# Melakukan prediksi
predictions = [knn.predict(train_features, train_labels, test_point) for test_point in test_features]

# Mengevaluasi hasil prediksi
evaluation = Evaluation()
metrics, confusion_matrix = evaluation.calculate_metrics(test_labels, predictions)

# Menampilkan hasil
print("Hasil Prediksi:", predictions)
print("Label Sebenarnya:", test_labels.tolist())
print("\nConfusion Matrix:")
for true_label, row in confusion_matrix.items():
    print(f"{true_label}: {row}")
print("\nMetrik Evaluasi:")
for label, label_metrics in metrics.items():
    print(f"Kelas '{label}':")
    for metric_name, value in label_metrics.items():
        print(f"  {metric_name}: {value:.2f}")
