import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Deteksi Anomali Aktivasi", layout="wide")

st.title("📡 Deteksi Anomali Aktivasi Pelanggan Telko")

uploaded_file = st.file_uploader("📁 Upload file CSV berisi data aktivasi", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data yang Diupload")
    st.dataframe(df.head())

    # --- Preprocessing & Feature Engineering ---
    with st.spinner("🔍 Mendeteksi anomali..."):
        df["AssignHour"] = pd.to_datetime(df["AssignTime"], errors='coerce').dt.hour
        df["ActivationHour"] = pd.to_datetime(df["Activation Time"], errors='coerce').dt.hour

        # Pilih fitur penting
        features = ["AssignHour", "ActivationHour", "Qty", "Duration", "Provider", "SKU"]
        df_features = df[features].copy()

        # Encode kolom kategorikal
        le = LabelEncoder()
        for col in ["Provider", "SKU"]:
            df_features[col] = le.fit_transform(df_features[col].astype(str))

        # Model
        model = IsolationForest(contamination=0.1, random_state=42)
        df["Anomaly"] = model.fit_predict(df_features)
        df["Anomaly_Label"] = df["Anomaly"].map({1: "✅ Normal", -1: "🚨 Anomali"})

    # --- Hasil ---
    st.success("✅ Deteksi selesai!")
    st.subheader("📋 Hasil Deteksi Anomali")
    st.dataframe(df[["Assignment ID", "Order ID", "AssignHour", "ActivationHour", "Qty", "Provider", "SKU", "Anomaly_Label"]])

    # Unduh hasil
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Hasil Deteksi (CSV)", data=csv, file_name="hasil_anomali.csv", mime="text/csv")
