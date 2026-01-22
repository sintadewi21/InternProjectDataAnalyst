@echo off
echo Menginstal dependensi baru (pyngrok)...
pip install -r requirements.txt
echo.
echo Menjalankan aplikasi dengan akses Publik...
echo Tekan Ctrl+C untuk berhenti.
python run_public.py
pause
