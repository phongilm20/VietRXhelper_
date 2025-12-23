"""
VietRx Vision Module: Handles real-time medication label detection.
Author: Xuan Thanh Phong Nguyen
Institution: Wright State University
"""

from ultralytics import YOLO
import easyocr
import cv2
import os

# Hyperparameters for Computer Vision Pipeline
MODEL_PATH = "best.pt"
YOLO_CONF_THRESHOLD = 0.45  # Filter weak detections to prevent UI lag
BOX_PADDING = 20           # Margin to ensure OCR captures the full text area

class VisionSystem:
    def __init__(self):
        """Initializes the YOLOv8 detector and the EasyOCR reader."""
        # Load custom YOLOv8 model for drug label localization
        self.detector = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        # Initialize EasyOCR (CPU mode for general compatibility)
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def extract_text_proposals(self, frame):
        """
        Processes a single frame to extract raw unstructured text.
        Args:
            frame: A numpy array representing the current webcam frame.
        Returns:
            list: A collection of detected text snippets.
        """
        if not self.detector or frame is None:
            return []
        
        # Phase 1: Object Detection (Localization)
        results = self.detector(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        candidate_list = []

        for r in results:
            for box in r.boxes:
                # Coordinate extraction
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w, _ = frame.shape
                
                # Image Cropping with boundary safety checks
                cropped = frame[max(0, y1-BOX_PADDING):min(h, y2+BOX_PADDING), 
                               max(0, x1-BOX_PADDING):min(w, x2+BOX_PADDING)]

                # Phase 2: Optical Character Recognition (OCR)
                ocr_data = self.reader.readtext(cropped)
                for (_, text, conf_ocr) in ocr_data:
                    if len(text) > 2:  # Filter noise
                        candidate_list.append({"text": text.strip()})
        return candidate_list