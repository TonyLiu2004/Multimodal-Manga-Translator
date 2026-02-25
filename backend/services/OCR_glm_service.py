from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from pathlib import Path
from helpers import get_project_root


class OCR_Glm_Service:
    def __init__(self, ocr_path=None, device=None):
        if not ocr_path:
            ROOT = get_project_root()
            ocr_path = ROOT / "backend" / "models" / "GlmOcr"
        processor_path = ocr_path / "processor"
        model_path = ocr_path / "model"

        if not processor_path.exists() or not model_path.exists():
            print(f"GLM OCR processor/model not found at {ocr_path}. Attempting to download")
            self.load_model()

        if processor_path.exists() and model_path.exists():
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, tie_word_embeddings=False)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print("Loaded glm OCR")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")

    def runOCR(self, image_url):
        # image = Image.open(image_url).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_url)}, 
                    {"type": "text", "text": "Text Recognition:"}
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        inputs.pop("token_type_ids", None)
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        output_text = self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_text
    
    def load_model(self):
        ROOT = get_project_root()
        GLMOCR_MODEL_DIR = ROOT / "backend" / "models" / "GlmOcr"
        MODEL_PATH = "zai-org/GLM-OCR"

        model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
        model.save_pretrained(GLMOCR_MODEL_DIR / "model")
        processor = AutoProcessor.from_pretrained( MODEL_PATH)
        processor.save_pretrained(GLMOCR_MODEL_DIR / "processor")

        print(f"Downloaded GLM OCR to: {GLMOCR_MODEL_DIR}")
        return str(GLMOCR_MODEL_DIR)