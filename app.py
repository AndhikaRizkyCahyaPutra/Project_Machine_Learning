from flask import Flask, request, jsonify, render_template
from data_preprocessing import DataPreprocessing
from knn import KNearestNeighbors
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Constants
DATA_FILE_PATH = "employee_dataset.csv"
TARGET_COLUMN_NAME = "leaveornot"

def normalize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df

def generate_confusion_matrix(y_true, y_pred, labels):
    """
    Generates a confusion matrix and returns it along with its heatmap as a base64 string.
    """
    # Map numeric labels to their string equivalents
    label_mapping = {0: "Bertahan", 1: "Resign"}
    labels_str = [label_mapping[label] for label in labels]  # Convert numeric labels to string

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_str, yticklabels=labels_str)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix untuk Prediksi Kelangsungan Kerja Karyawan\n Dalam 2 Tahun ke Depan.")

    # Save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()

    return cm, image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        print(f"Data diterima: {input_data}")
        input_df = pd.DataFrame([{
            'Education': input_data['education'],
            'JoiningYear': int(input_data['joiningYear']),
            'City': input_data['city'],
            'PaymentTier': int(input_data['paymentTier']),
            'Age': int(input_data['age']),
            'Gender': input_data['gender'],
            'EverBenched': input_data['everBenched'],
            'ExperienceInCurrentDomain': int(input_data['experienceInCurrentDomain'])
        }])

        input_df = normalize_columns(input_df)

        preprocessing = DataPreprocessing(DATA_FILE_PATH, target_column_name=TARGET_COLUMN_NAME)
        data_train = preprocessing.raw_data
        data_train = normalize_columns(data_train)
        print(data_train)
        data_combined = pd.concat([data_train, input_df], ignore_index=True)

        preprocessing.raw_data = data_combined
        data_encoded, target_encoded = preprocessing.preprocess()

        X_train = data_encoded.iloc[:-1, :].values
        y_train = target_encoded.iloc[:-1].values
        features = data_encoded.iloc[-1, :].values

        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print(f"Distribusi y_train setelah undersampling: {Counter(y_resampled)}")

        knn = KNearestNeighbors(k_neighbors=150)
        result = knn.classify(X_resampled, y_resampled, features)

        prediction = result["prediction"]
        nearest_neighbors = result["nearest_neighbors"]
        class_counts = result["class_counts"]

        # Ubah label 0 dan 1 menjadi 'Tidak' dan 'Ya'
        prediction_label = "Tidak" if prediction == 0 else "Ya"
        nearest_neighbors = [
            {**neighbor, "class": "Tidak" if neighbor["class"] == 0 else "Ya"}
            for neighbor in nearest_neighbors
        ]
        class_counts = {
            "Tidak": class_counts.get(0, 0),
            "Ya": class_counts.get(1, 0)
        }

        print(f"Hasil prediksi: {prediction_label}")
        print(f"Tetangga terdekat: {nearest_neighbors}")
        print(f"Jumlah kelas: {class_counts}")

        return jsonify({
            "prediction": prediction_label,
            "nearest_neighbors": nearest_neighbors,
            "class_counts": class_counts
        })
        
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

    
from sklearn.model_selection import KFold

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        input_data = request.json
        k_value = int(input_data["kValue"])  # Nilai K untuk KNN
        num_folds = 5  # Jumlah fold untuk k-fold validation

        # Preprocessing data
        preprocessing = DataPreprocessing(DATA_FILE_PATH, target_column_name=TARGET_COLUMN_NAME)
        data_train = preprocessing.raw_data
        data_train = normalize_columns(data_train)

        preprocessing.raw_data = data_train
        data_encoded, target_encoded = preprocessing.preprocess()

        features = data_encoded.values
        labels = target_encoded.values

        # K-Fold Cross Validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        precision_scores, recall_scores, accuracy_scores, f1_scores = [], [], [], []

        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Undersampling untuk menangani class imbalance
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

            # KNN Evaluation
            knn = KNearestNeighbors(k_neighbors=k_value)
            y_pred = [knn.classify(X_resampled, y_resampled, query_point)["prediction"] for query_point in X_test]

            # Calculate Metrics for the Current Fold
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # Calculate Average Metrics Across Folds
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_accuracy = np.mean(accuracy_scores)
        avg_f1 = np.mean(f1_scores)

        # Confusion Matrix (Optional: Generate for the Last Fold)
        labels_set = list(set(labels))
        cm, cm_image = generate_confusion_matrix(y_test, y_pred, labels_set)

        return jsonify({
            "precision": avg_precision,
            "recall": avg_recall,
            "accuracy": avg_accuracy,
            "f1Score": avg_f1,
            "confusionMatrix": cm_image
        })

    except Exception as e:
        print(f"Error saat evaluasi: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
