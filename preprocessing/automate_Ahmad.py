import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Memuat dataset mentah dari path yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {file_path}")
    print(f"[1/4] Memuat data dari {file_path}...")
    return pd.read_csv(file_path)

def clean_missing_values(df):
    """Menangani missing values tersembunyi pada kolom TotalCharges."""
    print("[2/4] Membersihkan missing values pada TotalCharges...")
    df_clean = df.copy()
    # Ubah spasi kosong menjadi NaN dan konversi ke float
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(" ", np.nan)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(float)
    
    # Imputasi dengan nilai median
    median_val = df_clean['TotalCharges'].median()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
    return df_clean

def preprocess_features(df):
    """Menghapus kolom yang tidak penting dan melakukan encoding."""
    print("[3/4] Melakukan seleksi fitur dan encoding...")
    df_proc = df.copy()
    
    # Hapus customerID jika ada
    if 'customerID' in df_proc.columns:
        df_proc = df_proc.drop(columns=['customerID'])
        
    # Encoding target variable 'Churn' (Yes -> 1, No -> 0)
    if 'Churn' in df_proc.columns:
        le = LabelEncoder()
        df_proc['Churn'] = le.fit_transform(df_proc['Churn'])
        
    # Pisahkan fitur dan target untuk One-Hot Encoding pada fitur saja
    target = df_proc['Churn']
    features = df_proc.drop(columns=['Churn'])
    
    # One-Hot Encoding fitur kategorikal
    features_encoded = pd.get_dummies(features, drop_first=True)
    features_encoded = features_encoded.astype(int)
    
    # Gabungkan kembali
    df_final = features_encoded.copy()
    df_final['Churn'] = target
    return df_final

def run_pipeline(input_path, output_path):
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    try:
        # 1. Load Data
        raw_data = load_data(input_path)
        
        # 2. Clean Data
        cleaned_data = clean_missing_values(raw_data)
        
        # 3. Preprocess Features
        final_data = preprocess_features(cleaned_data)
        
        # 4. Save Data
        print(f"[4/4] Menyimpan dataset bersih ke {output_path}...")
        final_data.to_csv(output_path, index=False)
        print("Pipeline Preprocessing Selesai dengan Sukses!")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat menjalankan pipeline: {str(e)}")

if __name__ == "__main__":
    # Path default untuk pengujian lokal di Colab
    INPUT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_FILE = "namadataset_preprocessing.csv"
    
    run_pipeline(INPUT_FILE, OUTPUT_FILE)