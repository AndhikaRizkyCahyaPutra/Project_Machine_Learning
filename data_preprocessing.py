import pandas as pd

class DataPreprocessing:
    def __init__(self, file_path, target_column_name):
        """
        Inisialisasi kelas dengan file CSV dan kolom target.
        :param file_path: Path ke file CSV.
        :param target_column_name: Nama kolom target.
        """
        self.file_path = file_path
        self.target_column_name = target_column_name
        self.raw_data = pd.read_csv(file_path)
        self.preprocessed_data = None
        self.target_data = None  # Atribut untuk menyimpan data kolom target

    def handle_missing_values(self):
        """Tangani nilai kosong dalam dataset dengan mengisi nilai kosong dengan 'None'."""
        print("Handling missing values...")
        self.raw_data.fillna('None', inplace=True)

    def separate_target_column(self):
        """Pisahkan kolom target dari data fitur dan simpan data target."""
        if self.target_column_name not in self.raw_data.columns:
            print(f"Kolom yang tersedia dalam dataset: {self.raw_data.columns.tolist()}")
            raise ValueError(f"Kolom target '{self.target_column_name}' tidak ditemukan dalam dataset.")
        # Pisahkan kolom target dan hapus dari data fitur
        print(f"Memisahkan kolom target: {self.target_column_name}")
        self.target_data = self.raw_data[self.target_column_name]
        self.raw_data.drop(columns=[self.target_column_name], inplace=True)

    def encode_categorical_features(self):
        """Lakukan one-hot encoding pada fitur kategorikal."""
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("Data belum diproses atau kosong. Pastikan metode sebelumnya sudah dijalankan.")
        print("Encoding categorical features...")
        categorical_columns = self.raw_data.select_dtypes(include=['object']).columns
        self.preprocessed_data = pd.get_dummies(self.raw_data, columns=categorical_columns, drop_first=True)

    def normalize_numerical_features(self):
        """Normalisasi fitur numerik agar berada dalam skala 0-1."""
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            raise ValueError("Data belum tersedia untuk normalisasi. Pastikan encoding telah dilakukan.")
        print("Normalizing numerical features...")
        numerical_columns = self.preprocessed_data.select_dtypes(include=['int64', 'float64']).columns
        for column in numerical_columns:
            min_value = self.preprocessed_data[column].min()
            max_value = self.preprocessed_data[column].max()
            if max_value != min_value:
                self.preprocessed_data[column] = (self.preprocessed_data[column] - min_value) / (max_value - min_value)
            else:
                self.preprocessed_data[column] = 0

    def preprocess(self):
        """Lakukan seluruh tahapan preprocessing pada dataset."""
        print("Starting preprocessing...")
        self.handle_missing_values()
        self.separate_target_column()
        self.encode_categorical_features()
        self.normalize_numerical_features()
        print("Preprocessing completed.")
        return self.preprocessed_data, self.target_data
