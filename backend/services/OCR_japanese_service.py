from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
from helpers import get_project_root
import os

ROOT = get_project_root()

class OCR_Japanese_Service:
    def __init__(self, ocr_path=None, device=None):
        ROOT = get_project_root()
        self.base_model_path = Path(os.getenv("MODEL_PATH", ROOT / "backend" / "models"))

        if not ocr_path:
            ocr_path = self.base_model_path / "Kha-white"
        else:
            ocr_path = Path(ocr_path)

        processor_path = ocr_path / "processor"
        model_path = ocr_path / "model"
        tokenizer_path = ocr_path / "tokenizer"

        if not processor_path.exists() or not model_path.exists() or not tokenizer_path.exists():
            print(f"Kha-white Japanese OCR model/processor/tokenizer not found at {ocr_path}. Attempting to download.")
            self.load_model()

        if processor_path.exists() and model_path.exists() and tokenizer_path.exists():
            self.processor = AutoImageProcessor.from_pretrained(processor_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, tie_word_embeddings=False)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print("Loaded Japanese OCR")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")
    
    def runOCR(self, image_url):
        image = Image.open(image_url).convert("L")

        inputs = self.processor(images=image, return_tensors="pt")
        generated_ids = self.model.generate(
            inputs.pixel_values, 
            # max_new_tokens=64,
            # num_beams=5,          
            # do_sample=False,     
            # early_stopping=True
        )

        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def load_model(self):
        DOWNLOAD_MODEL = "kha-white/manga-ocr-base"
        JAPANESE_OCR_DIR = self.base_model_path / "Kha-white"

        tokenizer = AutoTokenizer.from_pretrained(DOWNLOAD_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(DOWNLOAD_MODEL)
        processor = AutoImageProcessor.from_pretrained(DOWNLOAD_MODEL)

        tokenizer.save_pretrained(JAPANESE_OCR_DIR / "tokenizer")
        model.save_pretrained(JAPANESE_OCR_DIR / "model")
        processor.save_pretrained(JAPANESE_OCR_DIR / "processor")

        print(f"Downloaded Kha-white Japanese OCR to: {JAPANESE_OCR_DIR}")
        return str(JAPANESE_OCR_DIR)