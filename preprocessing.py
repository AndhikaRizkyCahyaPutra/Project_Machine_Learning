import pandas as pd

# Membaca file yang diunggah
file_path = 'enhanced_fever_medicine_recommendation.csv'
data = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama dari data
# print(data.head())

# Mengecek nilai yang hilang dalam dataset
missing_values = data.isnull().sum()

# Memproses fitur kategorikal dengan one-hot encoding
categorical_features = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Normalisasi fitur numerik (mengubah nilai ke skala 0-1)
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
data_encoded[numeric_features] = (data_encoded[numeric_features] - data_encoded[numeric_features].min()) / (
    data_encoded[numeric_features].max() - data_encoded[numeric_features].min())

# Menampilkan hasil preprocessing
print(data_encoded.head(), missing_values)
