from ultralytics import YOLO
from PIL import Image

class Bubble_Detector_Kiuyha_Service:
    def __init__(self, path):
        model_path = path / "kiuyha.pt"
        self.model = YOLO(model_path)
        print("Loaded Bubble Detector Kiuyha")

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
        row_height = img_h * 0.2 

        sorted_boxes = sorted(
            boxes_list, 
            key=lambda b: (
                (b['center_y'] // row_height),
                -b['center_x']             
            )
        )
        return sorted_boxes
    