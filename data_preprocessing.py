import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessing:
    def __init__(self, file_path, target_column_name):
        self.file_path = file_path
        self.target_column_name = target_column_name
        self.raw_data = pd.read_csv(file_path)
        self.preprocessed_data = None
        self.target_data = None  
        
    def handle_missing_values(self):
        print("Handling missing values...")
        for column in self.raw_data.columns:
            if self.raw_data[column].dtype == 'object':  
                self.raw_data[column].fillna('None', inplace=True)
            else:  
                self.raw_data[column].fillna(self.raw_data[column].mean(), inplace=True)

    def separate_target_column(self):
        if self.target_column_name not in self.raw_data.columns:
            print(f"Kolom yang tersedia dalam dataset: {self.raw_data.columns.tolist()}")
            raise ValueError(f"Kolom target '{self.target_column_name}' tidak ditemukan dalam dataset.")
        print(f"Memisahkan kolom target: {self.target_column_name}")
        self.target_data = self.raw_data[self.target_column_name]
        self.raw_data.drop(columns=[self.target_column_name], inplace=True)

    def encode_categorical_features(self):
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("Data belum diproses atau kosong. Pastikan metode sebelumnya sudah dijalankan.")
        print("Encoding categorical features...")
        categorical_columns = self.raw_data.select_dtypes(include=['object']).columns
        self.preprocessed_data = pd.get_dummies(self.raw_data, columns=categorical_columns, drop_first=True)

    def normalize_numerical_features(self):
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
                
    def validate_dataset(self):
        if self.raw_data.empty:
            raise ValueError("Dataset kosong. Pastikan file berisi data.")
        if self.target_column_name not in self.raw_data.columns:
            raise ValueError(f"Kolom target '{self.target_column_name}' tidak ditemukan dalam dataset.")
        print("Dataset validasi berhasil.")

    def visualize_correlation(self):
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            raise ValueError("Data belum tersedia untuk visualisasi. Pastikan preprocessing telah dilakukan.")
        
        print("Visualizing correlation matrix with target...")
        
        if self.target_data is None:
            raise ValueError("Kolom target tidak tersedia. Pastikan pemisahan target telah dilakukan.")
        
        data_with_target = self.preprocessed_data.copy()
        data_with_target[self.target_column_name] = self.target_data
        
        correlation_matrix = data_with_target.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
        plt.title('Korelasi Antar Fitur dan Target')
        plt.show()

    
    def preprocess(self):
        print("Starting preprocessing...")
        self.handle_missing_values()
        self.validate_dataset()
        self.separate_target_column()
        self.encode_categorical_features()
        self.normalize_numerical_features()
        print("Preprocessing completed.")
        return self.preprocessed_data, self.target_data