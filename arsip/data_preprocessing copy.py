import pandas as pd

class DataPreprocessing:
    def __init__(self, file_path):
        # Inisialisasi dengan file path untuk membaca dataset
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data_encoded = None

    def handle_missing_values(self):
        # Menangani nilai yang hilang (misalnya, menghapus baris dengan nilai yang hilang)
        self.data.dropna(inplace=True)

    def encode_categorical_features(self):
        # Mengonversi fitur kategorikal menjadi one-hot encoding
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.data_encoded = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)

    def convert_boolean_to_numeric(self):
        # Mengonversi fitur boolean menjadi nilai 0 dan 1
        boolean_columns = self.data_encoded.select_dtypes(include=['bool']).columns
        self.data_encoded[boolean_columns] = self.data_encoded[boolean_columns].astype(int)

    def normalize_numeric_features(self):
        # Normalisasi fitur numerik ke skala 0-1
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data_encoded[numeric_features] = (self.data_encoded[numeric_features] - self.data_encoded[numeric_features].min()) / (
            self.data_encoded[numeric_features].max() - self.data_encoded[numeric_features].min())

    def preprocess(self):
        # Menjalankan seluruh proses preprocessing
        self.handle_missing_values()
        self.encode_categorical_features()
        self.convert_boolean_to_numeric()
        self.normalize_numeric_features()
        return self.data_encoded
