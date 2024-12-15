import pandas as pd

class DataPreprocessing:
    def __init__(self, file_path, target_column):
        """
        Inisialisasi kelas dengan file CSV dan kolom target.
        :param file_path: Path ke file CSV.
        :param target_column: Nama kolom target.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.data = pd.read_csv(file_path)
        self.data_encoded = None
        self.target = None  # Atribut untuk menyimpan kolom target

    def handle_missing_values(self):
        """Tangani nilai kosong di dataset."""
        if 'Previous_Medication' in self.data.columns:
            self.data['Previous_Medication'].fillna('None', inplace=True)
        self.data.fillna(method='ffill', inplace=True)

    def separate_target(self):
        """Pisahkan kolom target dari data fitur."""
        if self.target_column not in self.data.columns:
            print(f"Kolom tersedia di dataset: {self.data.columns.tolist()}")
            raise ValueError(f"Kolom target '{self.target_column}' tidak ditemukan di dataset asli.")
        # Pisahkan target dan hapus dari data
        self.target = self.data[self.target_column]
        self.data.drop(columns=[self.target_column], inplace=True)
        print(f"Kolom target '{self.target_column}' dipisahkan.")

    def encode_categorical_features(self):
        """Lakukan one-hot encoding pada fitur kategorikal (tanpa target)."""
        categorical_features = self.data.select_dtypes(include=['object']).columns
        self.data_encoded = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)

    def normalize_numeric_features(self):
        """Normalisasi fitur numerik ke skala 0-1."""
        numeric_features = self.data_encoded.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_features:
            min_val = self.data_encoded[col].min()
            max_val = self.data_encoded[col].max()
            if max_val != min_val:
                self.data_encoded[col] = (self.data_encoded[col] - min_val) / (max_val - min_val)
            else:
                self.data_encoded[col] = 0

    def preprocess(self):
        """Jalankan seluruh proses preprocessing."""
        self.handle_missing_values()
        self.separate_target()  # Pisahkan kolom target sebelum encoding
        self.encode_categorical_features()
        self.normalize_numeric_features()
        return self.data_encoded, self.target
