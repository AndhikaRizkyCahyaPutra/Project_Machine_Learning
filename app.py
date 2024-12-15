from flask import Flask, request, jsonify, render_template
from data_preprocessing import DataPreprocessing
from knn import KNN
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

def normalize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df

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
            'temperature': float(input_data['temperature']),
            'fever_severity': input_data['feverSeverity'],
            'age': int(input_data['age']),
            'gender': input_data['gender'],
            'bmi': float(input_data['bmi']),
            'headache': input_data['headache'],
            'body_ache': input_data['bodyAche'],
            'fatigue': input_data['fatigue'],
            'chronic_conditions': input_data['chronicConditions'],
            'allergies': input_data['allergies'],
            'smoking_history': input_data['smokingHistory'],
            'alcohol_consumption': input_data['alcoholConsumption'],
            'humidity': float(input_data['humidity']),
            'aqi': float(input_data['aqi']),
            'physical_activity': input_data['physicalActivity'],
            'diet_type': input_data['dietType'],
            'heart_rate': float(input_data['heartRate']),
            'blood_pressure': input_data['bloodPressure'],
            'previous_medication': input_data['previousMedication']
        }])

        input_df = normalize_columns(input_df)

        preprocessing = DataPreprocessing("enhanced_fever_medicine_recommendation.csv", target_column_name="recommended_medication")
        data_train = preprocessing.raw_data
        data_train = normalize_columns(data_train)
        data_combined = pd.concat([data_train, input_df], ignore_index=True)

        preprocessing.raw_data = data_combined
        data_encoded, target_encoded = preprocessing.preprocess()

        X_train = data_encoded.iloc[:-1, :].values
        y_train = target_encoded.iloc[:-1].values
        features = data_encoded.iloc[-1, :].values

        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print(f"Distribusi y_train setelah undersampling: {Counter(y_resampled)}")

        knn = KNN(k_neighbors=150)
        result = knn.predict(X_resampled, y_resampled, features)

        prediction = result["prediction"]
        nearest_neighbors = result["nearest_neighbors"]
        class_counts = result["class_counts"]

        print(f"Hasil prediksi: {prediction}")
        print(f"Tetangga terdekat: {nearest_neighbors}")
        print(f"Jumlah kelas: {class_counts}")

        return jsonify({
            "recommended_medication": str(prediction),
            "nearest_neighbors": nearest_neighbors,
            "class_counts": class_counts
        })
        
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    
    
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        input_data = request.json
        k_value = int(input_data["kValue"])
        test_size = float(input_data["testSize"]) / 100.0

        preprocessing = DataPreprocessing("enhanced_fever_medicine_recommendation.csv", target_column_name="recommended_medication")
        data_train = preprocessing.raw_data
        data_train = normalize_columns(data_train) 

        preprocessing.raw_data = data_train
        data_encoded, target_encoded = preprocessing.preprocess()

        features = data_encoded.values
        labels = target_encoded.values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

        print(f"y_train: {y_train}")
        print(f"y_test: {y_test}")

        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print(f"Distribusi y_train setelah undersampling: {Counter(y_resampled)}")

        knn = KNN(k_neighbors=k_value)
        y_pred = [knn.predict(X_resampled, y_resampled, query_point) for query_point in X_test]

        y_pred_labels = [pred["prediction"] for pred in y_pred]

        print(f"y_pred: {y_pred_labels}")

        cm = confusion_matrix(y_test, y_pred_labels, labels=list(set(y_train)))
        precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(set(y_train)), yticklabels=list(set(y_train)))
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title(f"Confusion Matriks (k={k_value}, Ukuran Data Test ={test_size*100:.0f}%)")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        return jsonify({
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1Score": f1,
            "confusionMatrix": image_base64
        })

    except Exception as e:
        print(f"Error saat evaluasi: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
