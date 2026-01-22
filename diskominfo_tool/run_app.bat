@echo off
setlocal
title Diskominfo Data Tool Launcher

:: 1. Cek apakah Python terinstall
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak ditemukan!
    echo Harap install Python terlebih dahulu dan pastikan centang "Add Python to PATH".
    echo Hubungi pembuat program jika kesulitan.
    pause
    exit /b
)

:: 2. Cek/Install dependensi
echo [1/2] Mengecek kebutuhan sistem (Library)...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Gagal menginstall beberapa library. 
    echo Pastikan komputer terhubung ke Internet saat pertama kali menjalankan ini.
)

:: 3. Jalankan Streamlit
echo [2/2] Menjalankan Aplikasi...
echo Jika browser tidak terbuka otomatis, buka: http://localhost:8501
streamlit run app.py --headless.browser_auto_open true

pause
