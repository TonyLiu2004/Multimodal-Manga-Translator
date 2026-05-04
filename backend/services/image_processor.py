from services.bubble_detector_kiuyha_service import Bubble_Detector_Kiuyha_Service
from services.translate_qwen_service import Translate_Qwen_Service
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import re
import io
import db
import torch
from sqlmodel import Session
from db.models import Panels
from pathlib import Path
from helpers import get_project_root, setup_fonts
from manga_ocr import MangaOcr
import httpx

class ImageProcessor:
    def __init__(self, bubble_detector, ocr_model, translate_model):
        self.bubble_detector_model = bubble_detector
        self.ocr_model = ocr_model
        self.translate_model = translate_model

    async def download_and_process(self, image_url: str, language: str):
        # Create a temporary file that stays on disk until we close it
        # 'delete=False' is important because some ML models need the file to stay closed/flushed before they can read it.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            
            # Download
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                tmp.write(response.content)
                tmp.flush()

        try:
            results = self.process_image(temp_path, language)
            return results

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temp file: {temp_path}")

    def process_from_memory(self, image_bytes: bytes, image_url: str, chapter_id: int, page_number: int):
        # Convert raw bytes to PIL image
        img = Image.open(io.BytesIO(image_bytes))
        
        width,height = img.size
        
        # Bubble detection and OCR
        bubble_results = self.bubble_detector_model.predict(img)
        if not bubble_results:
            print(f"No bubbles detected for {image_url}")
            return []
        
        texts= []
        for bubble in bubble_results:
            crop = img.crop((bubble['x1'], bubble['y1'], bubble['x2'], bubble['y2']))
            # Run the OCR model on the crop
            text = self.ocr_model.predict(crop)
            texts.append(text)
        
        # Translation    
        try:
            print("Translating with cloud Qwen model...")
            translated = self.translate_model.translate_cloud(texts)
        except Exception as e:
            print("API translation failed with Qwen, falling back to local model...")
            translated = self.translate_model.translate(texts)
            
        final_results = []
        for i, (bubble, orig, trans) in enumerate(zip(bubble_results, texts, translated)):
            final_results.append({
                'original_text': orig,
                'translated_text': trans,
                'x1': bubble['x1'],
                'y1': bubble['y1'],
                'x2': bubble['x2'],
                'y2': bubble['y2'],
                'width': width,
                'height': height,
                'bubble_index': i
            })
        
        self.save_to_db(final_results, image_url, chapter_id, page_number)
        
    def save_to_db(self, results, url, chapter_id, page_number):
        engine = db.get_engine()
        with Session(engine) as session:
            for res in results:
                panel = Panels(
                    chapter_id=chapter_id,
                    page_number=page_number,
                    panel_url=url,
                    original_text=res['original_text'],
                    translated_text=res['translated_text'],
                    x1=res['x1'], y1=res['y1'], x2=res['x2'], y2=res['y2'],
                    width=res['width'], height=res['height'],
                    bubble_index=res['bubble_index']
                )
                session.add(panel)
            session.commit()
    
    def process_image(self, image_path, language):
        bubble_results = self.bubble_detector_model.predict(image_path)
        # print(f"bubble results: {bubble_results}")
        img = Image.open(image_path)
        width, height = img.size
        # draw = ImageDraw.Draw(img)

        texts = []
        coordinates={}
        i=0
        for box_data in bubble_results:
            coords = box_data['coords']
            # draw.rectangle(coords, outline="red", width=1)
            box_cropped = img.crop(coords)
            # box_cropped = upscale_for_ocr(box_cropped, scale=3)
            # box_cropped.show()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                box_cropped.save(f.name)
                temp_path = f.name

            text = ""
            try:
                text = self.ocr_model(box_cropped) 
            except Exception as e:
                print(f"text OCR failed for {i}")

            text = re.sub(r'[\n\r\u2028\u2029]+', ' ', text) #remove new lines
            texts.append({"id": i, "text": text})
            coordinates[i] = coords
            i+=1
        print(f'OCR Complete, total {len(texts)} bubbles.')

        #add translated text to manga image
        try:
            print("Translating with cloud Qwen model...")
            translated = self.translate_model.translate_cloud(texts)
        except Exception as e:
            print("API translation failed with Qwen, falling back to local model...")
            translated = self.translate_model.translate(texts)

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
                "width": width,
                "height": height,
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "original_text": original_text,
                "translated_text": translated_text,
            })

            ######### Code for drawing translated text onto manga panel directly) ###########

            #wipe the space
        #     draw.rectangle(coords, fill="white", outline="white")

        #     ROOT = get_project_root()
        #     env_font = os.getenv("FONT_PATH")
        #     if env_font:
        #         FONT_PATH = Path(env_font) / "NotoSansCJK.ttc"
        #     else:
        #         FONT_PATH = ROOT / "backend" / "fonts" / "NotoSansCJK.ttc"

        #     # 1. Calculate the best fit
        #     lines, best_font, final_size, line_h = fit_text_to_box(draw, translated_text, coords, FONT_PATH)

        #     # Calculate total height of the block
        #     total_h = line_h * len(lines)

        #     # Start_y adjusted for the block height relative to the box center
        #     start_y = coords[1] + ((coords[3] - coords[1]) - total_h) / 2

        #     # 3. Draw each line centered horizontally
        #     for line in lines:
        #         line = line.strip()
        #         if not line: continue

        #         # Horizontal Centering
        #         line_w = draw.textlength(line, font=best_font)
        #         start_x = coords[0] + ((coords[2] - coords[0]) - line_w) / 2

        #         draw.text((start_x, start_y), line, font=best_font, fill="black")
        #         start_y += line_h

        # img.show()
        return bubble_data #img, bubble_data
    

########Test code, keeping it here as reference. Remove later################
# def show_boxes(image_path):
#     result = bubble_detector_model.predict(image_path)
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)
#     for box in result.boxes:
#         # Get coordinates as a list of floats
#         coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
#         draw.rectangle(coords, outline="red", width=1)

#         # label
#         conf = box.conf[0].item()
#         box_cropped = img.crop(coords)
#         # box_cropped = upscale_for_ocr(box_cropped, scale=3)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
#             box_cropped.save(f.name)
#             temp_path = f.name
#         draw.text(
#             (coords[0], coords[1] - 10),
#             "b",
#             fill="red",
#             font=font
#         )
#     img.show()

# def get_wrapped_text(text, font, max_width):
#     lines = []
#     words = text.split(' ') # Split by words for English
#     current_line = []

#     for word in words:
#         # Check if adding the next word exceeds the width
#         test_line = ' '.join(current_line + [word])
#         # getlength() is more accurate than getbbox for text width
#         if font.getlength(test_line) <= max_width:
#             current_line.append(word)
#         else:
#             lines.append(' '.join(current_line))
#             current_line = [word]

#     lines.append(' '.join(current_line))
#     return lines

# def fit_text_to_box(draw, text, box_coords, font_path, padding=5, initial_size=40):
#     x1, y1, x2, y2 = box_coords

#     padding = padding
#     target_width = (x2 - x1) - (padding * 2)
#     target_height = (y2 - y1) - (padding * 2)

#     current_size = initial_size
#     lines = []

#     while current_size > 8:
#         # index=0 for Japanese, 1 for Korean in NotoSansCJK
#         font = ImageFont.truetype(font_path, size=current_size)
#         lines = get_wrapped_text(text, font, target_width)

#         # Use a more reliable line height measurement
#         # getbbox can be inconsistent; use font.size * constant for better leading
#         line_height = int(current_size * 1.2)
#         total_height = line_height * len(lines)

#         if total_height <= target_height:
#             break
#         current_size -= 2 # Step down by 2 for speed

#     return lines, font, current_size, line_height