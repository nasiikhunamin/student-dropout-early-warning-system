# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan tinggi yang sedang menghadapi tingkat *dropout* mahasiswa yang sangat tinggi, mencapai kurang lebih 32.1%. Hal ini membawa dampak negatif yang signifikan, baik dari segi kerugian finansial (kehilangan pendapatan *tuition fee*) maupun reputasi institusi di mata masyarakat dan calon mahasiswa baru. Untuk mengatasinya, institusi membutuhkan sebuah sistem *early warning* yang dapat mendeteksi probabilitas seorang mahasiswa untuk didropout.

### Permasalahan Bisnis
1. Bagaimana mengidentifikasi mahasiswa yang paling berisiko untuk *dropout* sedini mungkin?
2. Faktor-faktor utama apa saja (akademik, sosial-ekonomi, administratif) yang paling menentukan tingkat *dropout*?
3. Bagaimana menyajikan insight data ini ke dalam bentuk *dashboard monitoring* agar tim akademik, kemahasiswaan, dan keuangan dapat melakukan intervensi dengan cepat?

### Cakupan Proyek
Cakupan pengerjaan proyek ini meliputi:
- **Eksplorasi Data (EDA)**: Menganalisa karakteristik distribusi mahasiswa, nilai (GPA), dan status pembayaran.
- **Data Preparation**: Membersihkan data, melakukan *feature engineering* (seperti penghitungan *Risk Score*, *Approval rate*), dan membagi data latih.
- **Machine Learning Modeling**: Membangun Model **XGBoost Classifier**.
- **Pembuatan Aplikasi Web**: Membuat (*prototype*) aplikasi prediksi menggunakan **Streamlit**.
- **Business Dashboard**: Membuat visualisasi *dashboard* menggunakan **Metabase**.

### Persiapan

Sumber data: Dataset performa mahasiswa yang tersimpan di dalam berkas `dataset/data.csv`.

Setup environment:
```bash
# Instalasi seluruh library yang dibutuhkan
pip install -r requirements.txt
```

## Business Dashboard
Visualisasi bisnis dibuat menggunakan **Metabase**. 
Dashboard memuat beberapa visualisasi pemantauan utama:
- **Komposisi Mahasiswa:** Persentase kelulusan vs dropout (Pie Chart).
- **Analisis Finansial:** Dampak tunggakan biaya terhadap risiko dropout (Stacked Bar).
- **Performa Akademik:** Tren perbandingan rata-rata nilai semester (Line Chart).
- **Segmentasi Jurusan:** Program studi dengan tingkat risiko tertinggi (Row Chart).
- **Tabel Intervensi:** Daftar detail mahasiswa berisiko tinggi yang butuh penanganan segera.

menjalankan Metabase lokal via Docker:
```bash
docker run -d -p 3001:3000 --name datascience metabase/metabase:latest
```
akses `http://localhost:3001` di browser.
email: root@mail.com
password:root123

## Menjalankan Sistem Machine Learning
Prototype sistem Machine Learning dibangun menggunakan pustaka **Streamlit**. memuat 3 tampilan (Tab): Prediksi Individu (dilengkapi parameter input), Prediksi Batch massal berbasis *Upload CSV*, dan Tab *Dashboard Insight* ringkasan Exploratory Data Analysis.

Jalankan perintah berikut di direktori utama proyek:
```bash
streamlit run app.py
```
Aplikasi secara otomatis dapat diakses melalui browser pada `http://localhost:8501`.
Link Deploy: https://student-dropout-early-warning-system.streamlit.app

## Conclusion
Sistem deteksi dini berhasil dibangun secara akurat menggunakan **XGBoost Classifier** dengan performa **Macro F1-Score 0.716** (Kemampuan deteksi kelas Dropout mencapai **77.9%**).

Tiga temuan faktor utama penyebab *dropout*:
1. Riwayat performa akademik rendah (skor risiko tinggi & minimnya kelulusan mata kuliah).
2. Status tunggakan biaya kuliah (Tuition fee).
3. Tingkat absensi dan kegagalan mata kuliah (*Failures*).

### Rekomendasi Action Items
- **Monitoring Berkala:** Fokuskan sesi bimbingan khusus bagi mahasiswa yang terindikasi probabilitas *Dropout* > 70% di akhir semester satu.
- **Relaksasi Finansial:** Integrasikan tim akademik dan keuangan untuk memberi kelonggaran biaya (atau beasiswa) khusus bagi mahasiswa rawan yang menunggak cicilan.
- **Mentoring Wajib:** Wajibkan kelas perbaikan (*catch-up*) bagi mahasiswa yang memiliki persentase absen tinggi dan pernah gagal pada suatu mata kuliah (Failures > 0).
