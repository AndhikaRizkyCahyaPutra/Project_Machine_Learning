from flask import Flask, request, jsonify, render_template
from data_preprocessing import DataPreprocessing
from knn import KNN
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari request JSON
        input_data = request.json
        
        # Pastikan semua data input diterima
        print(f"Data diterima: {input_data}")

        # Proses data
        features = np.array([
            input_data['temperature'], input_data['age'], input_data['bmi'], 
            input_data['humidity'], input_data['aqi'], input_data['heartRate']
        ]).astype(float)  # Pastikan tipe data numerik
        
        features = np.random.rand(24).astype(float)
        # One-hot encode data kategorikal
        categorical_features = [
            input_data['feverSeverity'], input_data['gender'], 
            input_data['headache'], input_data['bodyAche']
        ]
        
        # Data preprocessing (simulasi preprocessing)
        preprocessing = DataPreprocessing("enhanced_fever_medicine_recommendation.csv")
        data_train = preprocessing.preprocess()  # Normalisasi & encoding dataset

        # Memisahkan fitur dan label
        X_train = data_train.iloc[:, :-1].values  # Semua kolom kecuali label
        y_train = data_train.iloc[:, -1].values   # Kolom label
        
        # Jalankan algoritma KNN
        knn = KNN(k_neighbors=5)
        prediction = knn.predict(X_train, y_train, features)
        
        # Kirimkan hasil ke frontend
        return jsonify({"recommended_medication": int(prediction)})
    
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
