from ultralytics import YOLO
from PIL import Image

class Bubble_Detector_Kiuyha_Service:
    def __init__(self, path):
        model_path = path / "kiuyha.pt"
        self.model = YOLO(model_path)

    def predict(self, img_path, conf=0.2, iou=0.4, show_labels=True, show_conf=True):
        results = self.model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            show_labels=show_labels,
            show_conf=show_conf,
        )
        return results[0]
    