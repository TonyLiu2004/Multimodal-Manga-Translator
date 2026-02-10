from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
from helpers import get_project_root

ROOT = get_project_root()

class OCR_Japanese_Service:
    def __init__(self, ocr_path=None, device=None):
        processor_path = ocr_path / "processor"
        model_path = ocr_path / "model"
        tokenizer_path = ocr_path / "tokenizer"

        self.processor = AutoImageProcessor.from_pretrained(processor_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path, tie_word_embeddings=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Loaded Japanese OCR")
    
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