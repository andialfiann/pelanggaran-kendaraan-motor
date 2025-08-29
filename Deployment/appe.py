import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai
import io
import re
from datetime import datetime
import streamlit as st
import pandas as pd

# ==============================================================================
# FUNGSI MEMUAT MODEL (dari utils.py)
# ==============================================================================
@st.cache_resource
def load_yolo(model_path="best5.pt"):
    """Memuat model deteksi YOLO."""
    # Harap pastikan file 'best5.pt' ada di direktori yang sama
    # atau berikan path yang benar.
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO dari path: {model_path}. Pastikan file ada. Error: {e}")
        return None


@st.cache_resource
def load_trocr(model_name='ziyadazz/OCR-PLAT-NOMOR-INDONESIA'):
    """Memuat model OCR TrOCR dan prosesornya."""
    try:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Gagal memuat model TrOCR. Periksa koneksi internet Anda. Error: {e}")
        return None, None

# ==============================================================================
# FUNGSI PEMROSESAN GAMBAR (dari utils.py)
# ==============================================================================
def preprocess_for_ocr(image_pil):
    """Membersihkan dan mempertajam gambar PIL untuk meningkatkan akurasi OCR."""
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = image_cv2.shape[:2]
    scale_factor = 3
    upscaled_image = cv2.resize(image_cv2, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_image_bgr = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)
    final_image_pil = Image.fromarray(cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB))
    return final_image_pil

def adjust_brightness(image_pil: Image.Image, factor: float) -> Image.Image:
    """Menyesuaikan kecerahan gambar PIL."""
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image

# ==============================================================================
# FUNGSI-FUNGSI OCR (dari utils.py)
# ==============================================================================
def run_trocr(processor, model, image_pil: Image.Image) -> str:
    """Menjalankan OCR menggunakan TrOCR pada sebuah gambar PIL."""
    if not processor or not model:
        return "Error: Model TrOCR tidak dimuat."
    try:
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return extracted_text
    except Exception as e:
        return f"Error TrOCR: {e}"

def run_gemini_ocr(api_key: str, image_pil: Image.Image) -> str:
    """Menjalankan OCR menggunakan Google AI Studio Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = "Transkripsikan semua teks dan angka yang terlihat di gambar ini. Hanya kembalikan teksnya saja, tanpa deskripsi tambahan."
        response = model.generate_content([prompt, image_pil])
        cleaned_text = re.sub(r'```|markdown', '', response.text)
        return cleaned_text.strip()
    except Exception as e:
        if "API_KEY_INVALID" in str(e):
            return "Error: API Key Gemini tidak valid."
        return "Error: Gagal menghubungi Gemini."

# ==============================================================================
# FUNGSI RANGKUMAN DENGAN AI (dari utils.py)
# ==============================================================================
def get_gemini_summary(api_key: str, violation_list: list) -> str:
    """Menghasilkan rangkuman UU atau ucapan selamat menggunakan Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        if not violation_list:
            prompt = "Buatkan ucapan selamat yang singkat dan ramah untuk pengendara motor yang patuh aturan lalu lintas di Indonesia. Sertakan ajakan untuk selalu menjadi contoh yang baik."
        else:
            violations_text = " dan ".join(violation_list)
            prompt = (f"Jelaskan secara singkat dalam satu paragraf mengenai sanksi dan pasal dalam UU LLAJ No. 22 Tahun 2009 untuk pelanggaran: '{violations_text}' saat berkendara motor di Indonesia. "
                      f"Setelah itu, berikan saran untuk membeli helm SNI dan sertakan tautan pencarian Google Maps dalam format URL lengkap untuk 'toko helm SNI terdekat' agar mudah divisualisasikan.")

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan rangkuman dari AI: {e}"

# ==============================================================================
# APLIKASI STREAMLIT
# ==============================================================================
st.set_page_config(page_title="🚦 Deteksi Pelanggaran", layout="wide")
st.title("🚦 Deteksi & Klasifikasi Pelanggaran Kendaraan Roda Dua")

# --- Memuat Model ---
yolo_model = load_yolo()
ocr_processor, ocr_model = load_trocr()

# --- Sidebar untuk API Key ---
st.sidebar.header("Konfigurasi API")
gemini_api_key = st.sidebar.text_input("Masukkan Google AI Studio API Key Anda", type="password")

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
        
    # Variabel untuk menyimpan hasil agar bisa diakses di luar kolom
    jenis_pelanggaran_list = []

    if predict_button:
        if not gemini_api_key:
            st.error("Harap masukkan API Key Gemini Anda di sidebar.")
        else:
            with st.spinner("Menganalisis gambar..."):
                results = yolo_model.predict(adjusted_img, verbose=False)

                img_with_boxes = results[0].plot()
                img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

                image_placeholder.image(img_with_boxes_rgb, caption="Hasil Deteksi", use_container_width=True)

                detection_rows = []
                for box in results[0].boxes:
                    conf = round(float(box.conf), 2)
                    cls_id = int(box.cls)
                    label_name = yolo_model.names.get(cls_id, 'unknown')
                    box_data = [int(x) for x in box.xyxy[0].tolist()]
                    detection_rows.append({"pred_box": box_data, "confidence": conf, "label": label_name})

                df = pd.DataFrame(detection_rows)

            with col2:
                deteksi_helm = "Pengendara menggunakan helm"
                plat_nomor_prediksi = "Tidak terdeteksi"
                tanggal_pajak_string = "Tidak terdeteksi"
                pajak_hidup_mati = "Tidak terdeteksi"

                if 'Non-Helmet' in df['label'].values:
                    deteksi_helm = "Pengendara tidak menggunakan helm"

                filtered_plate = df[df['label'] == 'License-Plate']
                if not filtered_plate.empty:
                    best_plate_row = filtered_plate.loc[filtered_plate['confidence'].idxmax()]
                    cropped_plate = adjusted_img.crop(best_plate_row['pred_box'])
                    processed_plate = preprocess_for_ocr(cropped_plate)
                    ocr_result = run_gemini_ocr(gemini_api_key, processed_plate)
                    
                    if "Error" in ocr_result or not ocr_result:
                        st.warning("OCR Gemini gagal untuk plat nomor, mencoba TrOCR...")
                        ocr_result = run_trocr(ocr_processor, ocr_model, processed_plate)
                    
                    date_match = re.search(r'(\b\d{2}[-.\s]+\d{2,4}\b)$', ocr_result)
                    if date_match:
                        tanggal_pajak_string = date_match.group(1).strip()
                        plat_nomor_prediksi = re.sub(r'(\b\d{2}[-.\s]+\d{2,4}\b)$', '', ocr_result).strip()
                    else:
                        plat_nomor_prediksi = ocr_result

                if tanggal_pajak_string == "Tidak terdeteksi":
                    filtered_exp_date = df[df['label'] == 'Exp-Date']
                    if not filtered_exp_date.empty:
                        best_exp_date_row = filtered_exp_date.loc[filtered_exp_date['confidence'].idxmax()]
                        cropped_exp_date = adjusted_img.crop(best_exp_date_row['pred_box'])
                        processed_exp_date = preprocess_for_ocr(cropped_exp_date)
                        ocr_result_date = run_gemini_ocr(gemini_api_key, processed_exp_date)
                        
                        if "Error" in ocr_result_date or not ocr_result_date:
                            st.warning("OCR Gemini gagal untuk tanggal, mencoba TrOCR...")
                            ocr_result_date = run_trocr(ocr_processor, ocr_model, processed_exp_date)
                        
                        if ocr_result_date and "Error" not in ocr_result_date:
                            tanggal_pajak_string = ocr_result_date.strip()

                if tanggal_pajak_string != "Tidak terdeteksi":
                    try:
                        cleaned_date = re.sub(r'[^0-9]', '', tanggal_pajak_string)
                        if len(cleaned_date) == 4:
                            exp_month = int(cleaned_date[:2])
                            exp_year = int(cleaned_date[2:]) + 2000
                            exp_date = datetime(exp_year, exp_month, 28)
                            
                            if exp_date > datetime.now():
                                pajak_hidup_mati = "Pajak motor hidup"
                            else:
                                pajak_hidup_mati = "Pajak motor kadaluwarsa"
                        else:
                            pajak_hidup_mati = f"Format tanggal tidak valid: {tanggal_pajak_string}"
                    except (ValueError, IndexError):
                        pajak_hidup_mati = f"Tidak bisa memproses tanggal: {tanggal_pajak_string}"

                if deteksi_helm == "Pengendara tidak menggunakan helm":
                    jenis_pelanggaran_list.append("tidak menggunakan helm")
                if pajak_hidup_mati == "Pajak motor kadaluwarsa":
                    jenis_pelanggaran_list.append("pajak motor mati")

                if not jenis_pelanggaran_list:
                    kelayakan = "Pengendara Layak"
                    jenis_pelanggaran = "Pengendara tidak melanggar aturan lalu lintas"
                else:
                    kelayakan = "Pengendara tidak layak"
                    jenis_pelanggaran = "Pengendara " + " dan ".join(jenis_pelanggaran_list)

                st.subheader("📊 Hasil Analisis")
                st.metric(label="Status Kelayakan", value=kelayakan)
                st.text(f"Jenis Pelanggaran: {jenis_pelanggaran}")
                st.text(f"Plat Nomor (OCR): {plat_nomor_prediksi}")
                st.text(f"Tanggal Exp (OCR): {tanggal_pajak_string}")
                st.text(f"Status Pajak: {pajak_hidup_mati}")
                st.divider()

                st.subheader("Detail Deteksi Objek")
                header_cols = st.columns([2, 2, 2])
                header_cols[0].markdown("Cropped Image")
                header_cols[1].markdown("Classes")
                header_cols[2].markdown("Confidence")
                st.divider()

                all_labels = ['Helmet', 'Non-Helmet', 'License-Plate'] + list(df[~df['label'].isin(['Helmet', 'Non-Helmet', 'License-Plate'])]['label'].unique())
                custom_order = pd.CategoricalDtype(all_labels, ordered=True)
                df['label_sorted'] = df['label'].astype(custom_order)
                sorted_df = df.sort_values(by=['label_sorted', 'confidence'], ascending=[True, False])

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

            # --- PERBAIKAN: Bagian Rangkuman AI dipindahkan ke luar kolom ---
            # Ini akan membuatnya ditampilkan di tengah dan lebar.
            st.divider()
            st.subheader("📝 Rangkuman Tambahan dari AI")
            with st.spinner("Memuat rangkuman dari AI..."):
                summary = get_gemini_summary(gemini_api_key, jenis_pelanggaran_list)
                if not jenis_pelanggaran_list:
                    st.success(f"*Pesan Kepatuhan:*\n\n{summary}")
                else:
                    st.info(f"*Informasi Peraturan:*\n\n{summary}")