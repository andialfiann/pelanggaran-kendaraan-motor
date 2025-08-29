import cv2
import numpy as np
import re
from datetime import datetime
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import pandas as pd
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai

# ==============================================================================
# FUNGSI MEMUAT MODEL
# ==============================================================================
def load_yolo(model_path="Deployment/best6.pt"): # Tambahkan "Deployment/"
    """Memuat model deteksi YOLO."""
    return YOLO(model_path)

def load_trocr(model_name='ziyadazz/OCR-PLAT-NOMOR-INDONESIA'):
    """Memuat model OCR TrOCR dan prosesornya."""
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

# ==============================================================================
# FUNGSI PEMROSESAN GAMBAR
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
    return enhancer.enhance(factor)

# ==============================================================================
# FUNGSI-FUNGSI OCR
# ==============================================================================
def run_trocr(processor, model, image_pil: Image.Image) -> str:
    """Menjalankan OCR menggunakan TrOCR pada sebuah gambar PIL."""
    try:
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Error TrOCR: {e}"

def run_gemini_ocr(api_key: str, image_pil: Image.Image) -> str:
    """Menjalankan OCR menggunakan Google AI Studio Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = "Transkripsikan semua teks dan angka yang terlihat di gambar ini. Hanya kembalikan teksnya saja, tanpa deskripsi tambahan."
        response = model.generate_content([prompt, image_pil])
        return re.sub(r'```|markdown', '', response.text).strip()
    except Exception as e:
        if "API_KEY_INVALID" in str(e): return "Error: API Key Gemini tidak valid."
        return "Error: Gagal menghubungi Gemini."

# ==============================================================================
# FUNGSI RANGKUMAN DENGAN AI
# ==============================================================================
def get_gemini_summary(api_key: str, violation_list: list) -> str:
    """Menghasilkan rangkuman UU atau ucapan selamat menggunakan Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        if not violation_list:
            prompt = "Anda adalah asisten keselamatan berkendara. Berikan pesan positif dan apresiasi kepada pengendara motor yang telah mematuhi aturan (memakai helm dan pajak hidup). Sampaikan pesan dengan gaya yang bersahabat dan memotivasi."
        else:
            violations_text = " dan ".join(violation_list)
            prompt = f"Anda adalah asisten hukum. Berikan rangkuman mengenai pelanggaran lalu lintas '{violations_text}'berdasarkan UU LLAJ No. 22 Tahun 2009. Jawaban harus mencakup dua poin utama:\n1. Dasar Hukum: Sebutkan pasal yang relevan.\n2. Sanksi: Jelaskan sanksi pidana (kurungan) dan denda maksimalnya."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan rangkuman dari AI: {e}"

# ==============================================================================
# FUNGSI UTAMA UNTUK ANALISIS (CONTROLLER)
# ==============================================================================
def analyze_image(image: Image.Image, yolo_model, ocr_processor, ocr_model, gemini_api_key):
    """Fungsi utama untuk menjalankan seluruh alur analisis."""
    
    # 1. Deteksi Objek
    results = yolo_model.predict(image, verbose=False)
    detection_rows = []
    for box in results[0].boxes:
        detection_rows.append({
            "pred_box": [int(x) for x in box.xyxy[0].tolist()],
            "confidence": round(float(box.conf), 2),
            "label": yolo_model.names.get(int(box.cls), 'unknown')
        })
    df = pd.DataFrame(detection_rows)
    
    # 2. Logika Analisis
    deteksi_helm = "Pengendara menggunakan helm"
    if 'Non-Helmet' in df['label'].values:
        deteksi_helm = "Pengendara tidak menggunakan helm"

    plat_nomor_prediksi = "Tidak terdeteksi"
    tanggal_pajak_string = "Tidak terdeteksi"
    pajak_hidup_mati = "Tidak terdeteksi"

    side_mirror_count = len(df[df['label'] == 'Side-Mirror'])

    filtered_plate = df[df['label'] == 'License-Plate']
    if not filtered_plate.empty:
        best_plate_row = filtered_plate.loc[filtered_plate['confidence'].idxmax()]
        cropped_plate = image.crop(best_plate_row['pred_box'])
        processed_plate = preprocess_for_ocr(cropped_plate)
        ocr_result = run_gemini_ocr(gemini_api_key, processed_plate)
        if "Error" in ocr_result or not ocr_result:
            ocr_result = run_trocr(ocr_processor, ocr_model, processed_plate)
        
        date_match = re.search(r'(\b\d{2}[-.\s]+\d{2,4}\b)$', ocr_result)
        if date_match:
            tanggal_pajak_string = date_match.group(1).strip()
            plat_nomor_prediksi = re.sub(r'(\b\d{2}[-.\s]+\d{2,4}\b)$', '', ocr_result).strip()
        else:
            plat_nomor_prediksi = ocr_result

    filtered_exp_date = df[df['label'] == 'Exp-Date']
    if not filtered_exp_date.empty and tanggal_pajak_string == "Tidak terdeteksi":
        best_exp_date_row = filtered_exp_date.loc[filtered_exp_date['confidence'].idxmax()]
        cropped_exp_date = image.crop(best_exp_date_row['pred_box'])
        processed_exp_date = preprocess_for_ocr(cropped_exp_date)
        ocr_result_date = run_gemini_ocr(gemini_api_key, processed_exp_date)
        if "Error" in ocr_result_date or not ocr_result_date:
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
                pajak_hidup_mati = "Pajak motor hidup" if exp_date > datetime.now() else "Pajak motor kadaluwarsa"
            else:
                pajak_hidup_mati = f"Format tanggal tidak valid: {tanggal_pajak_string}"
        except (ValueError, IndexError):
            pajak_hidup_mati = f"Tidak bisa memproses tanggal: {tanggal_pajak_string}"

    # 3. Penentuan Pelanggaran
    jenis_pelanggaran_list = []
    if deteksi_helm == "Pengendara tidak menggunakan helm":
        jenis_pelanggaran_list.append("tidak menggunakan helm")
    if pajak_hidup_mati == "Pajak motor kadaluwarsa":
        jenis_pelanggaran_list.append("pajak motor mati")
    if side_mirror_count < 1:
        jenis_pelanggaran_list.append("tidak memiliki kaca spion")
    if filtered_plate.empty:
        jenis_pelanggaran_list.append("plat nomor tidak terpasang")

    if not jenis_pelanggaran_list:
        kelayakan = "Pengendara Layak"
        jenis_pelanggaran = "Pengendara tidak melanggar aturan lalu lintas"
    else:
        kelayakan = "Pengendara tidak layak"
        jenis_pelanggaran = "Pengendara " + " dan ".join(jenis_pelanggaran_list)

    # 4. Gambar Hasil Anotasi
    img_with_boxes = results[0].plot()
    annotated_image = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    
    # 5. Kembalikan semua hasil dalam satu dictionary
    return {
        "annotated_image": annotated_image,
        "detections_df": df,
        "kelayakan": kelayakan,
        "jenis_pelanggaran": jenis_pelanggaran,
        "plat_nomor_ocr": plat_nomor_prediksi,
        "tanggal_pajak_ocr": tanggal_pajak_string,
        "status_pajak": pajak_hidup_mati,
        "violation_list": jenis_pelanggaran_list
    }
