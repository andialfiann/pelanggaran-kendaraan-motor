import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai
import io
import re

# ==============================================================================
# FUNGSI MEMUAT MODEL
# ==============================================================================
def load_yolo(model_path="best5.pt"):
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

def crop_image(img, box):
    """Memotong gambar berdasarkan koordinat bounding box."""
    return img.crop(box)

def adjust_brightness(image_pil: Image.Image, factor: float) -> Image.Image:
    """Menyesuaikan kecerahan gambar PIL."""
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image

# ==============================================================================
# FUNGSI-FUNGSI OCR
# ==============================================================================
def run_trocr(processor, model, image_pil: Image.Image) -> str:
    """Menjalankan OCR menggunakan TrOCR pada sebuah gambar PIL."""
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
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Menggunakan model vision terbaru
        prompt = "Transkripsikan semua teks dan angka yang terlihat di gambar ini. Hanya kembalikan teksnya saja, tanpa deskripsi tambahan."
        response = model.generate_content([prompt, image_pil])
        cleaned_text = re.sub(r'```|markdown', '', response.text)
        return cleaned_text.strip()
    except Exception as e:
        if "API_KEY_INVALID" in str(e):
            return "Error: API Key Gemini tidak valid."
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
            # Prompt jika tidak ada pelanggaran
            prompt = "Buatkan ucapan selamat yang singkat dan ramah untuk pengendara motor yang patuh aturan lalu lintas di Indonesia. Sertakan ajakan untuk selalu menjadi contoh yang baik."
        else:
            # Prompt jika ada pelanggaran
            violations_text = " dan ".join(violation_list)
            prompt = (f"Jelaskan secara singkat dalam satu paragraf mengenai sanksi dan pasal dalam UU LLAJ No. 22 Tahun 2009 untuk pelanggaran: '{violations_text}' saat berkendara motor di Indonesia. "
                      f"Setelah itu, berikan saran untuk membeli helm SNI dan sertakan tautan pencarian Google Maps dalam format URL lengkap untuk 'toko helm SNI terdekat' agar mudah divisualisasikan.")

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan rangkuman dari AI: {e}"

