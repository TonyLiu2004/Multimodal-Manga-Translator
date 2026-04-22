# from services.OCR_glm_service import OCR_Glm_Service
# from services.translate_tencentHY_service import Translate_Tencent_Service
# from services.bubble_detector_kitsumed_service import Bubble_Detector_Kitsumed_Service
from services.bubble_detector_kiuyha_service import Bubble_Detector_Kiuyha_Service
# from services.OCR_japanese_service import OCR_Japanese_Service
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
import uvicorn
from manga_ocr import MangaOcr
from fastapi.middleware.cors import CORSMiddleware
from api import router as manga_router
from fastapi.responses import JSONResponse
from services.image_processor import ImageProcessor

######################

app = FastAPI(
    title="Manga Translator API",
    description="Read endpoints for chapters and segments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Currently allow all origins, should be restricted to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(manga_router)

#####################

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

################################################

# GLMOCR_MODEL_DIR = MODEL_PATH / "GlmOcr"
# ocr_model = OCR_Glm_Service(GLMOCR_MODEL_DIR)

# JAPANESE_OCR_MODEL_DIR = MODEL_PATH / "Kha-white"
# ocr_japanese_model = OCR_Japanese_Service(JAPANESE_OCR_MODEL_DIR)

BUBBLE_DETECTOR_MODEL_DIR = MODEL_PATH / "kiuyha.pt"
bubble_detector_model = Bubble_Detector_Kiuyha_Service(BUBBLE_DETECTOR_MODEL_DIR)

# cn_translate_model = Translate_Tencent_Service()

translate_model = Translate_Qwen_Service()

ocr_model = MangaOcr()

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

processor = ImageProcessor(bubble_detector_model, ocr_model, translate_model)
print("Finished loading all models and fonts")

def show_boxes(image_path):
    result = bubble_detector_model.predict(image_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    print(result)
    for box in result:
        # Get coordinates as a list of floats
        coords = box['coords']#.xyxy[0].tolist() # [x1, y1, x2, y2]
        draw.rectangle(coords, outline="red", width=1)

        # label
        # conf = box.conf[0].item()
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

###
###
###

@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    """Return 400 for invalid provider_id etc."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.post("/test")
def test(img_path: Optional[str] = None):
    print("test called")
    if not img_path:
        img_path = "./test_2.png"
    img_path = Path(img_path)
    print(f"image path: {img_path}")
    if img_path.exists():
        bubble_data = processor.process_image(img_path, "japanese")
        print(bubble_data)
        return {"result": bubble_data}
    else:
        print(f"{img_path} does not exist")

@app.post("/api/manga/translate")
async def translate_manga_panel(image_url: str, language: str = ""):
    try:
        results = await processor.download_and_process(image_url, language)
        return {"status": "success", "data": results}    
    except Exception as e:
        print(f"Translation Route Error: {e}")
    
if __name__ == "__main__":
    # processor.process_image("../test_images/test_3.png", "")
    # show_boxes("../test_images/test_3.png")
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Starting Production Server on Port {port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)