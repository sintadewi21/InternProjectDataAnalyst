import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    """
    Memuat data dari file CSV atau Excel yang diunggah.
    
    Args:
        uploaded_file: File object yang diunggah oleh user via Streamlit.
        
    Returns:
        pandas.DataFrame: DataFrame berisi data yang dimuat, atau None jika gagal.
    """
    if uploaded_file is None:
        return None
        
    try:
        file_name = uploaded_file.name
        
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
            return None
            
        return clean_data(df)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        return None

def clean_data(df):
    """
    Membersihkan dan mengkonversi tipe data kolom secara otomatis.
    Khususnya menangani format angka Indonesia (ribuan='.', desimal=',')
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                cleaned_col = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                
                converted_col = pd.to_numeric(cleaned_col, errors='coerce')
                
                if converted_col.notna().mean() > 0.5:
                    df[col] = converted_col
            except:
                continue
                
        if df[col].dtype == 'object':
            try:
                converted_col = pd.to_numeric(df[col], errors='coerce')
                if converted_col.notna().mean() > 0.5:
                    df[col] = converted_col
            except:
                pass
                
    return df
