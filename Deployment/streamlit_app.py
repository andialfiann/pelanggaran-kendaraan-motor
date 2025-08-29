from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ==== YOLO Functions ====
def load_yolo(model_path="best4.pt"):
    return YOLO(model_path)

def detect_objects(model, image):
    results = model(image)
    return results

# ==== TrOCR Functions ====
def load_trocr(model_name="microsoft/trocr-base-stage1"):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

def predict_text(processor, model, cropped_img: Image.Image):
    pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ==== Utils ====
def crop_image(img, box):
    return img.crop(box)

def format_label(label: str):
    if label.lower() == "helmet":
        return f"<span style='color:green; font-weight:bold'>{label}</span>"
    elif label.lower() == "non-helmet":
        return f"<span style='color:red; font-weight:bold'>{label}</span>"
    return label
