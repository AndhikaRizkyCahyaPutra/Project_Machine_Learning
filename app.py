from flask import Flask, request, jsonify, render_template
from data_preprocessing import DataPreprocessing
from knn import KNN  # Menggunakan KNN yang sudah Anda buat
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

def normalize_columns(df):
    """Normalisasi nama kolom agar konsisten"""
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df

@app.route('/')
def index():
    return render_template('index.html')  # Render halaman HTML utama

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Mendapatkan data dari request JSON
        input_data = request.json
        print(f"Data diterima: {input_data}")  # Debugging input

        # 2. Buat dataframe sementara untuk data test
        input_df = pd.DataFrame([{
            'Temperature': float(input_data['temperature']),
            'Fever_Severity': input_data['feverSeverity'],
            'Age': int(input_data['age']),
            'Gender': input_data['gender'],
            'BMI': float(input_data['bmi']),
            'Headache': input_data['headache'],
            'Body_Ache': input_data['bodyAche'],
            'Fatigue': input_data['fatigue'],
            'Chronic_Conditions': input_data['chronicConditions'],
            'Allergies': input_data['allergies'],
            'Smoking_History': input_data['smokingHistory'],
            'Alcohol_Consumption': input_data['alcoholConsumption'],
            'Humidity': float(input_data['humidity']),
            'AQI': float(input_data['aqi']),
            'Physical_Activity': input_data['physicalActivity'],
            'Diet_Type': input_data['dietType'],
            'Heart_Rate': float(input_data['heartRate']),
            'Blood_Pressure': input_data['bloodPressure'],
            'Previous_Medication': input_data['previousMedication']
        }])

        # Normalisasi nama kolom data test
        input_df = normalize_columns(input_df)

        # 3. Gabungkan data test dengan data training
        preprocessing = DataPreprocessing("enhanced_fever_medicine_recommendation.csv")
        data_train = preprocessing.data
        data_train = normalize_columns(data_train)  # Normalisasi nama kolom data training
        data_combined = pd.concat([data_train, input_df], ignore_index=True)

        # 4. Lakukan preprocessing pada data gabungan
        preprocessing.data = data_combined
        data_encoded = preprocessing.preprocess()

        # Debugging data setelah preprocessing
        print(f"Data setelah preprocessing: {data_encoded.shape}")
        print(f"Kolom data setelah preprocessing: {data_encoded.columns.tolist()}")

        # 5. Pisahkan kembali data training dan data test
        X_train = data_encoded.iloc[:-1, :-1].values  # Semua baris kecuali terakhir
        y_train = data_encoded.iloc[:-1, -1].values   # Kolom terakhir (label)
        features = data_encoded.iloc[-1, :-1].values  # Baris terakhir (data test)

        # Debugging dimensi data dan distribusi awal
        print(f"Dimensi X_train: {X_train.shape}")
        print(f"Dimensi y_train: {y_train.shape}")
        print(f"Distribusi y_train sebelum balancing: {Counter(y_train)}")

        # Undersampling pada kelas mayoritas (kelas 0)
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print(f"Distribusi y_train setelah undersampling: {Counter(y_resampled)}")

        # 6. Prediksi menggunakan KNN
        knn = KNN(k_neighbors=150)  # Membuat objek KNN dengan k_neighbors = 5
        prediction = knn.predict(X_resampled, y_resampled, features)  # Mengirimkan data uji (features)

        # Debugging hasil prediksi
        print(f"Hasil prediksi: {prediction}")

        return jsonify({"recommended_medication": str(prediction)})

    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
