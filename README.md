# Sistem Prediksi Loyalitas Karyawan dengan KNN

## Deskripsi Project
Sistem ini fungsinya untuk memprediksi apakah seorang karyawan akan bertahan atau resign dari perusahaan dalam 2 tahun ke depan. Project ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi berdasarkan data karyawan. Sistem ini memiliki interface web menggunakan **Flask**, sehingga user bisa melakukan prediksi dan evaluasi model secara interaktif.

---

## Fitur Utama
1. **Prediksi Loyalitas Karyawan:**
   - lakukan input informasi karyawan untuk memprediksi apakah mereka akan bertahan atau resign.
   - Hasil prediksi adalah berupa label "0" atau "1", yang mana 0 mewakili bertahan dan 1 mewakili resign. selain itu juga ada  informasi tentang tetangga terdekat.

2. **Evaluasi Model KNN:**
   - Pada program ini dilakukan proses *K-Fold Cross Validation* untuk menentukan nilai `k` terbaik.
   - Terdapat metric evaluasi seperti **Precision**, **Recall**, **Accuracy**, dan **F1-Score**.
   - Visualisasi *Confusion Matrix*.

3. **Interface Web:**
   - Halaman input data karyawan.
   - Halaman evaluasi model.

---

## Teknologi yang Digunakan
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, Bootstrap
- **Machine Learning:** K-Nearest Neighbors (KNN)
- **Visualisasi:** Matplotlib, Seaborn

---

## Instalasi dan Konfigurasi

### Requirement dasar
Pastikan sudah memiliki beberapa hal berikut ini:
1. Python 3.8 atau yang lebih baru
2. Paket manajer `pip`

### Langkah-langkah Install
1. Clone repositori proyek:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Buat dan aktifkan *virtual environment* (OPTIONAL):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate   # Untuk Windows
   ```

3. Instal dependensi dengan cepat:
   ```bash
   pip install -r requirements.txt
   ```

4. Pastikan file dataset ada di tempat yang benar:
   ```
   employee_dataset.csv
   ```

5. Run aplikasi:
   ```bash
   python app.py
   ```

6. Akses aplikasi lewat browser:
   ```
   http://127.0.0.1:5000
   ```

---

## Struktur Proyek
Berikut adalah struktur file proyek:

```
project_root/
├── app.py                   # File utama untuk menjalankan aplikasi Flask
├── knn.py                   # Implementasi algoritma K-Nearest Neighbors
├── data_preprocessing.py    # Modul preprocessing data
├── templates/
│   ├── index.html           # Halaman utama untuk prediksi
│   └── evaluation.html      # Halaman evaluasi model
├── employee_dataset.csv     # Dataset karyawan
├── requirements.txt         # Daftar dependensi
└── README.md                # Dokumentasi proyek
```

---

## Penjelasan Fungsi tiap File

### 1. **`app.py`**
File utama buat menjalankan aplikasi Flask. Fitur utama:

- **Endpoint `/`**
  - untuk Menampilkan halaman prediksi karyawan.

- **Endpoint `/evaluation`**
  - untuk Menampilkan halaman evaluasi model.

- **Endpoint `/predict`**
  - Menerima data karyawan sebagai input JSON.
  - Preprocessing data baru.
  - memproses algoritma KNN untuk prediksi.
  - return hasil prediksi berupa:
    - Kategori "Bertahan" atau "Resign".
    - Distribusi tetangga terdekat.

- **Endpoint `/evaluate`**
  - Melakukan evaluasi model menggunakan K-Fold Cross Validation.
  - return berupa metrik evaluasi seperti Precision, Recall, Accuracy, dan F1-Score.

---

### 2. **`knn.py`**
File ini Berisikan implementasi algoritma K-Nearest Neighbors (KNN):

- **Fungsi `calculate_distance`:**
  Menghitung Euclidean distance antara dua titik.

- **Fungsi `find_nearest_neighbors`:**
  Menentukan tetangga terdekat pada data input.

- **Fungsi `determine_majority_class`:**
  Menentukan kelas mayoritas dari k tetangga terdekat.

- **Fungsi `classify`:**
  Melakukan klasifikasi berdasarkan hasil voting tetangga terdekat.

---

### 3. **`data_preprocessing.py`**
File ini berisikan langkah-langkah preprocessing data:

- **Handling Missing Values:**
  Mengisi nilai kosong pada suatu kolom dengan rata-rata (numerik) atau nilai default (kategorikal).

- **One-Hot Encoding:**
  Mengubah fitur kategorikal menjadi bentuk numerik.

- **Normalisasi:**
  Menskalakan data numerik ke rentang [0, 1].

---

### 4. **Template HTML**

#### a. **`index.html`**
Halaman ini berguna untuk prediksi karyawan:
- Form input data karyawan.
- Setelah data diproses, maka akan tampil tabel hasil prediksi, tetangga terdekat, dan distribusi kelas.

#### b. **`evaluation.html`**
Halaman ini berguna untuk evaluasi model:
- Button untuk start proses evaluasi
- Menampilkan metrik evaluasi dan confusion matrix.

---

## Cara Penggunaan

### a. Prediksi Karyawan
1. Buka program aplikasi di browser.
2. Isi seluruh data pada form dengan data karyawan baru
3. Klik **Submit** untuk memproses dan menampilkan hasil prediksi.
4. Hasil yang ditampilkan:
   - Apakah karyawan "Bertahan" atau "Resign".
   - Tabel tetangga terdekat dan distribusi kelas.

### b. Evaluasi Model
1. Buka halaman **KNN Evaluasi** dari button di halaman utama.
2. Klik button **Evaluasi Model**.
3. Tunggu program memproses hingga evaluasi selesai.
4. Hasil yang ditampilkan:
   - Nilai `k` terbaik.
   - Tabel nilai `k` dan F1-Score.
   - Metrik evaluasi (Precision, Recall, Accuracy, F1-Score).
   - Visualisasi confusion matrix.

---

## Dependensi
File `requirements.txt` berisi:

```
Flask==2.3.2
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.2
imbalanced-learn==0.11.0
matplotlib==3.8.2
seaborn==0.12.2
```

Instal semua dependensi menggunakan:
```bash
pip install -r requirements.txt
```

---

## Dataset
File dataset `employee_dataset.csv` harus ada di direktori project. Kolom yang digunakan adalah:
- **Education**: Pendidikan terakhir.
- **JoiningYear**: Tahun bergabung di perusahaan.
- **City**: Kota tempat bekerja.
- **PaymentTier**: Tingkat gaji (1: Tinggi, 2: Menengah, 3: Rendah).
- **Age**: Usia karyawan.
- **Gender**: Jenis kelamin.
- **EverBenched**: Apakah pernah menganggur (No Project).
- **ExperienceInCurrentDomain**: Pengalaman di bidang saat ini (tahun).
- **LeaveOrNot**: Kolom target (0: Bertahan, 1: Resign).

---
