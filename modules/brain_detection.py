import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class BrainTumorDetector:
    def __init__(self, model_path="Models/brain-model-2.pt"):
        self.model = None
        self.load_model(model_path)

    def load_model(self, path):
        if os.path.exists(path):
            self.model = YOLO(path)
            print("[Brain] Model loaded.")
        else:
            print(f"[Brain] Warning: Model file not found at {path}")

    def predict(self, img_pil):
        if not self.model:
            return "Model Missing", 0.0, None, None, "Model file not found.", 0.0

        # Convert PIL to CV2 (RGB)
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # Needed for processing logic
        h, w = img_rgb.shape[:2]

        # YOLO Inference
        results = self.model.predict(source=img_rgb, conf=0.25, verbose=False)
        res = results[0]
        
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        classes = res.boxes.cls.cpu().numpy().tolist()
        confs = res.boxes.conf.cpu().numpy().tolist()
        names = res.names

        annotated_img = img_rgb.copy()
        final_mask = np.zeros((h, w), dtype=np.uint8)
        detected_objects = []

        for i, box in enumerate(boxes):
            label = names[int(classes[i])]
            confidence = confs[i]

            detected_objects.append({"label": label, "confidence": confidence})

            if "no_tumor" in label.lower(): continue

            # Draw Bounding Box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Segmentation Logic (Project 1 Feature)
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_enhanced = clahe.apply(roi_gray)
                _, thresh = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], clean)

        # Statistics
        total_px = h * w
        tumor_px = cv2.countNonZero(final_mask)
        size_pct = (tumor_px / total_px) * 100

        # Classification Logic
        valid_tumors = [d for d in detected_objects if d['label'] != 'No Tumor']
        if valid_tumors:
            valid_tumors.sort(key=lambda x: x['confidence'], reverse=True)
            final_label = valid_tumors[0]['label']
            final_conf = valid_tumors[0]['confidence'] * 100
            explanation = "AI detected a tumor anomaly. Radiological verification recommended."
        else:
            final_label = "No Tumor"
            final_conf = 100.0
            explanation = "No tumor anomalies detected."

        return final_label, final_conf, Image.fromarray(annotated_img), Image.fromarray(final_mask) if tumor_px > 0 else None, explanation, size_pct