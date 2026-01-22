
import os
import sys
from pyngrok import ngrok
from streamlit.web import cli as stcli

# KONFIGURASI DOMAIN
# Jika Anda sudah mengklaim domain statis di dashboard ngrok (https://dashboard.ngrok.com/cloud-edge/domains),
# masukkan nama domain tersebut di bawah ini (dalam tanda kutip).
# Contoh: STATIC_DOMAIN = "tools-diskominfo.ngrok-free.app"
# Jika dibiarkan None, ngrok akan memberikan domain acak setiap kali dijalankan.
STATIC_DOMAIN = None 

def main():
    # Periksa apakah token autentikasi ngrok sudah diatur
    # Jika belum, pyngrok mungkin akan gagal atau menggunakan sesi terbatas
    print("Mempersiapkan public URL...")
    
    try:
        # Membuka tunnel ke port 8501
        if STATIC_DOMAIN:
            public_url = ngrok.connect(8501, domain=STATIC_DOMAIN).public_url
        else:
            public_url = ngrok.connect(8501).public_url
            
        print(f"\n========================================================")
        print(f"  PUBLIC URL APP ANDA: {public_url}")
        print(f"========================================================\n")
    except Exception as e:
        print(f"Gagal membuat public URL: {e}")
        print("Pastikan Anda sudah login ke ngrok (ngrok config add-authtoken <token>) jika diperlukan.")
        return

    # Menjalankan Streamlit
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
