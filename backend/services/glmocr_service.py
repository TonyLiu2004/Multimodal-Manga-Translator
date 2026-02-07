from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch
import os
from pathlib import Path

class OCR_Service:
    def __init__(self, ocr_path=None, device= None):
        processor_path = ocr_path / "processor"
        model_path = ocr_path / "model"

        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
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