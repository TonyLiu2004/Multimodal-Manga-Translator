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
import torch
from pathlib import Path
from helpers import get_project_root, setup_fonts
from fastapi import FastAPI
from typing import Optional
import db as manga_db

###
###
###

ROOT = get_project_root()

#Path base and fallbacks
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "backend" / "models"))

#Font path
env_font = os.getenv("FONT_PATH")
if env_font:
    FONT_PATH = Path(env_font) / "NotoSansCJK.ttc"
else:
    FONT_PATH = ROOT / "backend" / "fonts" / "NotoSansCJK.ttc"

# Device defaults to 'cpu' if not specified
env_device = os.getenv("DEVICE", "cpu").lower()
if env_device in ["amd", "cuda"]:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device_name = "cpu"

device = torch.device(device_name)

print(f"Loading models from {MODEL_PATH} and fonts from {FONT_PATH}")

app = FastAPI()

#####################

GLMOCR_MODEL_DIR = MODEL_PATH / "GlmOcr"
ocr_model = OCR_Glm_Service(GLMOCR_MODEL_DIR)

JAPANESE_OCR_MODEL_DIR = MODEL_PATH / "Kha-white"
ocr_japanese_model = OCR_Japanese_Service(JAPANESE_OCR_MODEL_DIR)

BUBBLE_DETECTOR_MODEL_DIR = MODEL_PATH / "kiuyha.pt"
bubble_detector_model = Bubble_Detector_Kiuyha_Service(BUBBLE_DETECTOR_MODEL_DIR)

cn_translate_model = Translate_Tencent_Service()

translate_model = Translate_Qwen_Service()


if not FONT_PATH.exists():
    print(f"Font NotoSansCJK not found at {FONT_PATH}. Attempting to download.")
    setup_fonts()

if FONT_PATH.exists():
    font = ImageFont.truetype(
        FONT_PATH,
        size=12,
        index=7
    )
else:
    raise FileNotFoundError(f"Font NotoSansCJK not found at {FONT_PATH}")

print("Finished loading all models and fonts")

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
    print(f'OCR Complete, total {len(texts)} bubbles.')

    #add translated text to manga image
    try:
        print("Translating with cloud Qwen model...")
        translated = translate_model.translate_cloud(texts)
    except Exception as e:
        print("API translation failed with Qwen, falling back to local model...")
        translated = translate_model.translate(texts)

    print(translated)

    bubble_data = []
    for i in range(len(texts)):
        coords = coordinates[i]
        x1, y1, x2, y2 = coords
        original_text = texts[i]["text"]
        translated_text = translated.get(str(i), translated.get(i, ""))
        if not isinstance(translated_text, str):
            translated_text = str(translated_text)
        print(f"{i}: {original_text}")
        print(translated_text)
        print("==================================")

        bubble_data.append({
            "bubble_index": i,
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "original_text": original_text,
            "translated_text": translated_text,
        })

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

    return img, bubble_data

def translate_text(text, language):
    # translated_text = ""
    # if language == "japanese":
    #     translated_text =

    translated_text = translate_model.translate(text)

    return translated_text

def runOCRTests(image_url):
    text = ocr_model.runOCR(image_url)

def _language_to_code(language: str) -> str:
    """Map language name to ISO 639-1 style code for DB."""
    m = {"japanese": "ja", "english": "en", "korean": "ko", "chinese": "zh"}
    return m.get(language.lower(), language[:2] if len(language) >= 2 else "ja")


def process_chapter(
    manga_title: str,
    chapter_number: float,
    page_paths: list,
    language: str = "japanese",
    provider_id: str = "local",
    db_url: str = None,
):
    """
    Process each page of a chapter, draw translated text on images, and save
    to the PostgreSQL text repository (provider_id, manga_title, chapter/page,
    segment coordinates, original/translated text, language code). No images stored.
    page_paths: list of paths to page images in order.
    provider_id: source/provider identifier (e.g. 'mangadex', 'local').
    db_url: PostgreSQL URL or set DATABASE_URL.
    Returns (list of (img, bubble_data) per page).
    """
    manga_db.init_db(db_url)
    language_code = _language_to_code(language)
    results = []
    for page_number, image_path in enumerate(page_paths, start=1):
        path = Path(image_path)
        if not path.exists():
            print(f"Skip missing page {page_number}: {path}")
            continue
        print(f"Processing chapter {chapter_number} page {page_number}/{len(page_paths)}: {path.name}")
        img, bubble_data = process_image(str(path), language)
        manga_db.save_page_translation(
            provider_id=provider_id,
            manga_title=manga_title,
            chapter_number=chapter_number,
            page_number=page_number,
            bubbles=bubble_data,
            language_code=language_code,
            db_url=db_url,
        )
        results.append((img, bubble_data))
    print(f"Chapter '{manga_title}' ch.{chapter_number} saved to DB ({len(results)} pages).")
    return results


def main():
    img_path = "./test_2.png"
    img, bubble_data = process_image(img_path, "japanese")
    print(bubble_data)
    img.show()
    # manga_db.save_page_translation(provider_id="local", manga_title="Test", chapter_number=0,
    #     page_number=1, bubbles=bubble_data, language_code="ja")


@app.get("/")
def home():
    return {"status": "Manga Translator Backend Running"}

@app.post("/translate")
def translate_manga (data: dict):
    print(data)
    return {"result": "translated text"}

@app.post("/")
def test(img_path: Optional[str] = None):
    if not img_path:
        img_path = "./test_2.png"
    img_path = Path(img_path)

    if img_path.exists():
        img, bubble_data = process_image(img_path, "japanese")
        print(bubble_data)
        return {"result": bubble_data}
    else:
        print(f"{img_path} does not exist")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)