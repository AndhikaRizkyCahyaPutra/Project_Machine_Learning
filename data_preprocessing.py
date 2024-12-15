import pandas as pd

class DataPreprocessing:
    def __init__(self, file_path):
        # Inisialisasi dengan file path untuk membaca dataset
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data_encoded = None

    def handle_missing_values(self):
        """Menangani nilai yang hilang (missing values)"""
        print(f"Sebelum menangani nilai kosong: {self.data.shape}")
        print(f"Jumlah nilai kosong per kolom sebelum diisi:")
        print(self.data.isnull().sum())

        # Logika khusus untuk kolom `Previous_Medication`
        if 'Previous_Medication' in self.data.columns:
            self.data['Previous_Medication'].fillna('None', inplace=True)  # Isi dengan 'None'

        # Untuk kolom lainnya, gunakan forward fill
        self.data.fillna(method='ffill', inplace=True)

        print(f"Setelah menangani nilai kosong: {self.data.shape}")
        print(f"Jumlah nilai kosong per kolom setelah diisi:")
        print(self.data.isnull().sum())

    def encode_categorical_features(self):
        """Mengonversi fitur kategorikal menjadi one-hot encoding"""
        print(f"Sebelum one-hot encoding: {self.data.shape}")
        print(f"Kolom-kolom sebelum encoding: {self.data.columns.tolist()}")  # Cetak kolom
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.data_encoded = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)
        print(f"Setelah one-hot encoding: {self.data_encoded.shape}")
        print(f"Kolom-kolom setelah encoding: {self.data_encoded.columns.tolist()}")  # Cetak kolom

    def convert_boolean_to_numeric(self):
        """Mengonversi fitur boolean menjadi nilai 0 dan 1"""
        boolean_columns = self.data_encoded.select_dtypes(include=['bool']).columns
        print(f"Kolom boolean sebelum konversi: {boolean_columns.tolist()}")  # Cetak kolom boolean
        self.data_encoded[boolean_columns] = self.data_encoded[boolean_columns].astype(int)
        print(f"Setelah konversi boolean ke numerik: {self.data_encoded.shape}")
        print(f"Kolom setelah konversi boolean ke numerik: {self.data_encoded.columns.tolist()}")  # Cetak kolom

    def normalize_numeric_features(self):
        """Normalisasi fitur numerik ke skala 0-1"""
        numeric_features = self.data_encoded.select_dtypes(include=['int64', 'float64']).columns
        print(f"Kolom numerik sebelum normalisasi: {numeric_features.tolist()}")  # Cetak kolom numerik
        for col in numeric_features:
            min_val = self.data_encoded[col].min()
            max_val = self.data_encoded[col].max()
            if max_val != min_val:  # Cegah pembagian dengan nol
                self.data_encoded[col] = (self.data_encoded[col] - min_val) / (max_val - min_val)
            else:
                self.data_encoded[col] = 0  # Jika max dan min sama, atur ke 0
        print(f"Setelah normalisasi: {self.data_encoded.shape}")
        print(f"Kolom-kolom setelah normalisasi: {self.data_encoded.columns.tolist()}")  # Cetak kolom

    def preprocess(self):
        """Menjalankan seluruh proses preprocessing"""
        print(f"Kolom-kolom awal: {self.data.columns.tolist()}")  # Cetak kolom awal
        self.handle_missing_values()
        self.encode_categorical_features()
        self.convert_boolean_to_numeric()
        self.normalize_numeric_features()
        return self.data_encoded
