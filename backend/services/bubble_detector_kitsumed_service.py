from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class Bubble_Detector_Kitsumed_Service:
    def __init__(self, path):
        model_path = path / "kitsumed.pt"
        self.model = YOLO(model_path)
        print("Loaded Bubble Detector Kitsumed")

    def predict(self, img_path, conf=0.2, iou=0.4, show_labels=True, show_conf=True, imgsz=640):
        img = Image.open(img_path)
        results = self.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            show_labels=show_labels,
            show_conf=show_conf,
            imgsz=imgsz,
        )
        return results[0]
    