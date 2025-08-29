import streamlit as st
import pandas as pd
from PIL import Image
import os
from dotenv import load_dotenv # Import library dotenv

# Import fungsi utama dan fungsi pembantu dari logic.py
from logic import load_yolo, load_trocr, adjust_brightness, analyze_image, get_gemini_summary

# --- PERBAIKAN: Muat variabel dari file .env ---
load_dotenv()
# ---------------------------------------------

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="🚦 Deteksi Pelanggaran", layout="wide")
st.title("🚦 Deteksi & Klasifikasi Pelanggaran Kendaraan Roda Dua")

# Muat semua model menggunakan cache
@st.cache_resource
def load_all_models():
    yolo = load_yolo()
    processor, trocr = load_trocr()
    return yolo, processor, trocr

yolo_model, ocr_processor, ocr_model = load_all_models()

# --- PERBAIKAN: Baca API Key dari environment variable ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
# -------------------------------------------------------

# --- Sidebar Info Model ---
if yolo_model:
    st.sidebar.header("Informasi Model")
    st.sidebar.write("Kelas yang terdeteksi oleh model:")
    st.sidebar.json(yolo_model.names)

# --- Upload File ---
uploaded_file = st.file_uploader("Upload gambar kendaraan Anda di sini", type=["jpg", "jpeg", "png"])

if uploaded_file and yolo_model:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("📷 Pratinjau & Pengaturan Gambar")
        brightness_factor = st.slider("Atur Kecerahan", 0.2, 2.0, 1.0, 0.1)
        adjusted_img = adjust_brightness(original_img, brightness_factor)
        
        image_placeholder = st.empty()
        image_placeholder.image(adjusted_img, caption="Gambar yang akan dianalisis", use_container_width=True) 
        
        predict_button = st.button("🚀 Lakukan Prediksi", use_container_width=True)
    
    # Inisialisasi variabel untuk menyimpan hasil di luar scope kolom
    analysis_results = {}

    if predict_button:
        # --- PERBAIKAN: Cek apakah API key berhasil dimuat dari .env ---
        if not gemini_api_key:
            st.error("API Key Gemini tidak ditemukan. Pastikan Anda sudah membuat file .env dan mengisinya dengan benar.")
        else:
            with st.spinner("Menganalisis gambar..."):
                # Panggil fungsi analisis utama dari logic.py
                analysis_results = analyze_image(adjusted_img, yolo_model, ocr_processor, ocr_model, gemini_api_key)
                
                # Tampilkan gambar yang sudah dianotasi
                image_placeholder.image(analysis_results["annotated_image"], caption="Hasil Deteksi", use_container_width=True)

            with col2:
                # Tampilkan hasil analisis
                st.subheader("📊 Hasil Analisis")
                st.metric(label="Status Kelayakan", value=analysis_results["kelayakan"])
                st.text(f"Jenis Pelanggaran: {analysis_results['jenis_pelanggaran']}")
                st.text(f"Plat Nomor (OCR): {analysis_results['plat_nomor_ocr']}")
                st.text(f"Tanggal Exp (OCR): {analysis_results['tanggal_pajak_ocr']}")
                st.text(f"Status Pajak: {analysis_results['status_pajak']}")
                st.divider()

                st.subheader("Detail Deteksi Objek")
                header_cols = st.columns([2, 2, 2])
                header_cols[0].markdown("Cropped Image")
                header_cols[1].markdown("Classes")
                header_cols[2].markdown("Confidence")
                st.divider()

                df_detections = analysis_results["detections_df"]

                # Definisikan urutan kelas yang diinginkan
                all_known_labels = ['Helmet', 'Non-Helmet', 'License-Plate', 'Exp-Date', 'Side-Mirror']
                custom_order = pd.CategoricalDtype(all_known_labels, ordered=True)
                
                df_detections['label_sorted'] = df_detections['label'].astype(custom_order)
                sorted_df = df_detections.sort_values(
                    by=['label_sorted', 'confidence'], 
                    ascending=[True, False]
                )

                # Loop melalui DataFrame yang sudah diurutkan
                for index, row in sorted_df.iterrows():
                    row_cols = st.columns([2, 2, 2])
                    cropped_img = adjusted_img.crop(row['pred_box'])
                    row_cols[0].image(cropped_img, width=150)
                    label_name = row['label']
                    if "Helmet" == label_name:
                        row_cols[1].markdown(f"<p style='color:green; font-weight:bold;'>{label_name}</p>", unsafe_allow_html=True)
                    elif "Non-Helmet" == label_name:
                        row_cols[1].markdown(f"<p style='color:red; font-weight:bold;'>{label_name}</p>", unsafe_allow_html=True)
                    else:
                        row_cols[1].write(label_name)
                    row_cols[2].write(f"{row['confidence']:.2f}")
                    st.divider()

    # Rangkuman AI ditampilkan di luar dan di bawah kolom
    if predict_button and analysis_results:
        st.divider()
        st.subheader("📝 Rangkuman Tambahan dari AI")
        if not gemini_api_key:
            st.warning("Rangkuman AI tidak dapat ditampilkan karena API Key tidak ditemukan.")
        else:
            with st.spinner("Memuat rangkuman dari AI..."):
                summary = get_gemini_summary(gemini_api_key, analysis_results["violation_list"])
                if not analysis_results["violation_list"]:
                    st.success(f"*Pesan Kepatuhan:*\n\n{summary}")
                else:
                    st.info(f"*Informasi Peraturan:*\n\n{summary}")