# Panduan Instalasi & Penggunaan Diskominfo Data Tool

Aplikasi ini digunakan untuk analisis data statistik (Regresi, Forecasting, dll).
Berikut adalah cara untuk menjalankan aplikasi ini di komputer/laptop lain.

## 1. Persiapan Awal (Wajib)
Pastikan di komputer target sudah terinstall **Python**.
Jika belum, download dan install dari: https://www.python.org/downloads/
> **PENTING:** Saat install, centang kotak **"Add Python to PATH"** di bagian bawah installer.

## 2. Cara Menjalankan Aplikasi
1.  Copy seluruh folder ini ke komputer staff/target.
2.  Buka folder tersebut.
3.  Cari file bernama **`Buka_Aplikasi.vbs`** (ikonnya biasanya kertas biru/putih).
4.  **Klik 2x** file tersebut.
5.  Website akan otomatis terbuka di browser Anda (Chrome/Edge). CMD akan berjalan tersembunyi di latar belakang sehingga desktop tetap rapi.

## 4. Troubleshooting (Jika Terjadi Error)
Jika saat klik `Buka_Aplikasi.vbs` website tidak muncul:
1.  Buka folder, lalu klik file **`run_app.bat`** (ini akan memunculkan jendela hitam).
2.  Baca pesan error yang muncul di sana.
3.  **Error umum**: 
    - *Python not found*: Staff belum install Python atau lupa centang "Add to PATH".
    - *No module named '...'*: Pastikan komputer tersambung internet saat pertama kali menjalankan agar library bisa terdownload otomatis.
