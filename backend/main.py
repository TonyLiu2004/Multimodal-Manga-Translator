from services.OCR_glm_service import OCR_Glm_Service
from services.translate_tencentHY_service import Translate_Tencent_Service
from services.bubble_detector_kitsumed_service import Bubble_Detector_Kitsumed_Service
from services.bubble_detector_kiuyha_service import Bubble_Detector_Kiuyha_Service
from services.OCR_japanese_service import OCR_Japanese_Service
from services.translate_qwen_service import Translate_Qwen_Service
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import re
from pathlib import Path
from helpers import get_project_root

from manga_ocr import MangaOcr
mocr = MangaOcr()
###
###
###

ROOT = get_project_root()

GLMOCR_MODEL_DIR = ROOT / "backend" / "models" / "GlmOcr"
ocr_model = OCR_Glm_Service(GLMOCR_MODEL_DIR)

JAPANESE_OCR_MODEL_DIR = ROOT / "backend" / "models" / "Kha-white"
ocr_japanese_model = OCR_Japanese_Service(JAPANESE_OCR_MODEL_DIR)

CN_TRANSLATE_MODEL_DIR = ROOT / "backend" / "models" / "TencentHY"
cn_translate_model = Translate_Tencent_Service(CN_TRANSLATE_MODEL_DIR)

TRANSLATE_MODEL_DIR = ROOT / "backend" / "models" / "Qwen"
translate_model = Translate_Qwen_Service(TRANSLATE_MODEL_DIR)

BUBBLE_DETECTOR_MODLE_DIR = ROOT / "backend" / "models"
bubble_detector_model = Bubble_Detector_Kiuyha_Service(BUBBLE_DETECTOR_MODLE_DIR)



FONT_PATH = ROOT / "backend" / "fonts" / "NotoSansCJK.ttc"
font = ImageFont.truetype(
    FONT_PATH,
    size=12,
    index=7
)

###
###
###

def show_boxes(image_path):
    result = bubble_detector_model.predict(image_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in result.boxes:
        # Get coordinates as a list of floats
        coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        draw.rectangle(coords, outline="red", width=1)
        
        # label
        conf = box.conf[0].item()
        box_cropped = img.crop(coords)
        # box_cropped = upscale_for_ocr(box_cropped, scale=3)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            box_cropped.save(f.name)      
            temp_path = f.name
        draw.text(
            (coords[0], coords[1] - 10), 
            "b",  
            fill="red", 
            font=font
        )
    img.show()

def get_wrapped_text(text, font, max_width):
    lines = []
    words = text.split(' ') # Split by words for English
    current_line = []

    for word in words:
        # Check if adding the next word exceeds the width
        test_line = ' '.join(current_line + [word])
        # getlength() is more accurate than getbbox for text width
        if font.getlength(test_line) <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    lines.append(' '.join(current_line))
    return lines

def fit_text_to_box(draw, text, box_coords, font_path, padding=5, initial_size=40):
    x1, y1, x2, y2 = box_coords
    
    padding = padding
    target_width = (x2 - x1) - (padding * 2)
    target_height = (y2 - y1) - (padding * 2)
    
    current_size = initial_size
    lines = []
    
    while current_size > 8:
        # index=0 for Japanese, 1 for Korean in NotoSansCJK
        font = ImageFont.truetype(font_path, size=current_size)
        lines = get_wrapped_text(text, font, target_width)
        
        # Use a more reliable line height measurement
        # getbbox can be inconsistent; use font.size * constant for better leading
        line_height = int(current_size * 1.2) 
        total_height = line_height * len(lines)
        
        if total_height <= target_height:
            break
        current_size -= 2 # Step down by 2 for speed
        
    return lines, font, current_size, line_height

def upscale_for_ocr(img, scale=2):
    w, h = img.size
    return img.resize((w*scale, h*scale), Image.BICUBIC)

def process_image(image_path, language):
    bubble_results = bubble_detector_model.predict(image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    texts = []
    coordinates={}
    i=0
    for box_data in bubble_results:
        coords = box_data['coords']
        draw.rectangle(coords, outline="red", width=1)
        box_cropped = img.crop(coords)
        # box_cropped = upscale_for_ocr(box_cropped, scale=3)
        # box_cropped.show()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            box_cropped.save(f.name)      
            temp_path = f.name

        text = ""
        if language == "japanese":
            text = ocr_japanese_model.runOCR(temp_path)
            # text = mocr(temp_path)
        else:
            text = ocr_model.runOCR(temp_path)

        text = re.sub(r'[\n\r\u2028\u2029]+', ' ', text) #remove new lines
        texts.append({"id": i, "text": text})
        coordinates[i] = coords
        i+=1

        print(text)
        print("==================================")
    
    #add translated text to manga image
    print("TRANSLATING:")
    translated = translate_model.translate(texts)
    print(translated)
    for id, translated_text in translated.items():
        coords = coordinates[id]
        # print(f"{id}: {translated_text}")
        #wipe the space
        draw.rectangle(coords, fill="white", outline="white")

        # 1. Calculate the best fit
        lines, best_font, final_size, line_h = fit_text_to_box(draw, translated_text, coords, FONT_PATH)

        # Calculate total height of the block
        total_h = line_h * len(lines)

        # Start_y adjusted for the block height relative to the box center
        start_y = coords[1] + ((coords[3] - coords[1]) - total_h) / 2

        # 3. Draw each line centered horizontally
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Horizontal Centering
            line_w = draw.textlength(line, font=best_font)
            start_x = coords[0] + ((coords[2] - coords[0]) - line_w) / 2
            
            draw.text((start_x, start_y), line, font=best_font, fill="black")
            start_y += line_h

    return img

def translate_text(text, language):
    # translated_text = ""
    # if language == "japanese":
    #     translated_text = 

    translated_text = translate_model.translate(text)
          
    return translated_text

def runOCRTests():
    test_dir = ROOT / "test_images"
    for i in range(1, 10):
        try:
            image_url = test_dir / f"test_{i}.png"
            text = ocr_model.runOCR(image_url)
            print(f"=============={i}==============")
            print(text)
        except:
            print(f"failed on {i}")
            break

###
### testing
###

from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-ko-en"
kr_tokenizer = MarianTokenizer.from_pretrained(model_name)
kr_model = MarianMTModel.from_pretrained(model_name)

def kr_translate(text):
    text = [text]
    inputs = kr_tokenizer(text, return_tensors="pt", padding=True)
    outputs = kr_model.generate(**inputs, max_new_tokens=200)

    return (kr_tokenizer.batch_decode(outputs, skip_special_tokens=True))
###
###
###
def main():
    img_path = ROOT / "test_images" / "test_7.png"
    img = process_image(img_path, "japanese")
    img.show()

if __name__ == "__main__":
    main()