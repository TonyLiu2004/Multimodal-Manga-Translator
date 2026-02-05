from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#download OCR model locally
GLMOCR_MODEL_DIR = BASE_DIR / "backend" / "models" / "GlmOcr"
MODEL_PATH = "zai-org/GLM-OCR"
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
model.save_pretrained(GLMOCR_MODEL_DIR / "model")
processor = AutoProcessor.from_pretrained( MODEL_PATH)
processor.save_pretrained(GLMOCR_MODEL_DIR / "processor")