import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk halaman Home
def home():
    st.title("Selamat Datang di Aplikasi Prediksi Penggunaan Obat")
    st.write("""
    ### Author: Agil Ilham Putra
    Aplikasi ini dibuat untuk memprediksi penggunaan obat berdasarkan beberapa parameter seperti usia, jenis kelamin, tekanan darah, kolesterol, dan rasio Na/K.
    Anda dapat memasukkan data pasien di halaman Prediksi dan mendapatkan hasil prediksi secara langsung.
    """)

# Fungsi untuk halaman Prediksi
def prediksi():
    st.title("Prediksi Penggunaan Obat")
    
    # Input data dari pengguna
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    bp = st.selectbox("BP", options=[0, 1, 2], format_func=lambda x: ["Low", "Normal", "High"][x])
    cholesterol = st.selectbox("Cholesterol", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "High")
    na_to_k = st.number_input("Na_to_K", min_value=0.0, max_value=50.0, value=4.5)

    # Data input sebagai dictionary
    testing = {
        'Age': [age],
        'Sex': [sex],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'Na_to_K': [na_to_k]
    }

    # Konversi data input ke DataFrame
    testing_df = pd.DataFrame(testing)

    # Memuat model dari file
    try:
        knn = joblib.load('predict_drug_model.pkl')

        # Melakukan prediksi
        if st.button("Prediksi"):
            pred_coba = knn.predict(testing_df)
            st.write("Hasil Prediksi dari Pasien Baru:", pred_coba[0])
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Prediksi"])

# Menampilkan halaman yang dipilih
if page == "Home":
    home()
else:
    prediksi()
