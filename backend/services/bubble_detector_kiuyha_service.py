from ultralytics import YOLO
from PIL import Image
from helpers import get_project_root
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

class Bubble_Detector_Kiuyha_Service:
    def __init__(self, model_path=None):
        ROOT = get_project_root()
        self.base_model_path = Path(os.getenv("MODEL_PATH", ROOT / "backend" / "models"))

        if not model_path:
            model_path = self.base_model_path / "kiuyha.pt"
        else:
            model_path = Path(model_path)
            
        if not model_path.exists():
            print(f"Kiuyha model not found at {model_path}. Attempting to download")
            self.load_model()

        if model_path.exists():
            self.model = YOLO(model_path, task="detect") #'task=detect', 'segment', 'classify','pose' or 'obb'
            print("Loaded Bubble Detector Kiuyha")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")
          
    def predict(self, img_path, conf=0.2, iou=0.4, show_labels=True, show_conf=True, imgsz=640):
        results = self.model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            show_labels=show_labels,
            show_conf=show_conf,
            imgsz=imgsz
        )

        img_w, img_h = Image.open(img_path).size
        padding = 4
        boxes_list = []
        for box in results[0].boxes:  
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            pad_x1 = max(0, x1 - padding)
            pad_y1 = max(0, y1 - padding)
            pad_x2 = min(img_w, x2 + padding)
            pad_y2 = min(img_h, y2 + padding)
        
            boxes_list.append({
                'coords': [pad_x1, pad_y1, pad_x2, pad_y2],
                'center_x': (pad_x1 + pad_x2) / 2,
                'center_y': (pad_y1 + pad_y2) / 2
            })

        #sort right to left, top to bottom. test more
        row_height = img_h * 0.1 

        sorted_boxes = sorted(
            boxes_list, 
            key=lambda b: (
                (b['center_y'] // row_height),
                -b['center_x']             
            )
        )
        return sorted_boxes
    
    def load_model(self):
        target_path = self.base_model_path / "kiuyha.pt"

        if target_path.exists():
            print(f"Kiuya Model already exists at {target_path}")
            return str(target_path)
        
        downloaded_path = hf_hub_download(
            repo_id="Kiuyha/Manga-Bubble-YOLO",
            filename="weights/yolo26n.pt", #"model.pt",
            local_dir=self.base_model_path
        )

        final_path = Path(downloaded_path).rename(target_path)
        print(f"Downloaded Kiuyha bubble detector to: {final_path}")
        return str(final_path)