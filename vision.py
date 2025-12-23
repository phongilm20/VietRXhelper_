from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import os

# CONFIGURATION
MODEL_PATH = "best.pt"  # Custom trained model
CONFIDENCE_THRESHOLD = 0.4

print("[INFO] Initializing Vision System...")

# 1. Load Custom YOLO Model
try:
    if os.path.exists(MODEL_PATH):
        detector = YOLO(MODEL_PATH)
        print(f"[SUCCESS] Loaded custom model: {MODEL_PATH}")
    else:
        print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
        print("Please ensure best.pt is in the project directory.")
        detector = None
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    detector = None

# 2. Load OCR Engine
try:
    # Load English for drug names (Standard characters)
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception as e:
    print(f"[ERROR] Failed to load OCR: {e}")
    reader = None

BAD_WORDS = {
    "tablet", "tablets",
    "capsule", "capsules",
    "oral", "oral use"
}

def analyze_image(image_path):
    """
    Pipeline: Detect (YOLO) -> Crop -> OCR -> Text
    Returns: The detected drug name string (best guess).
    """
    if detector is None or reader is None:
        return ""

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return ""

    # Step 1: Object Detection
    results = detector(img, conf=CONFIDENCE_THRESHOLD, verbose=False)

    best_text = ""
    best_score = 0.0

    for r in results:
        for box in r.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Step 2: Image Cropping (with padding)
            h, w, _ = img.shape
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop_img = img[y1:y2, x1:x2]

            # Step 3: Text Recognition (OCR)
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            ocr_result = reader.readtext(gray_crop, detail=0)
            text = " ".join(ocr_result).strip()
            if len(text) < 3:
                continue

            t_low = text.lower().strip()
            # Remove general words
            if t_low in BAD_WORDS:
                continue

            # Scoring: conf + heuristics
            score = conf

            if "mg" in t_low or "mcg" in t_low:
                score += 0.3

            first = text.split()[0]
            if len(first) > 4:
                score += 0.2

            if score > best_score:
                best_score = score
                best_text = text

    if best_text:
        return best_text

    # Fallback: Full image scan if YOLO misses
    print("[WARNING] No strong object match. Scanning full image...")
    full_ocr = reader.readtext(img, detail=0)
    return " ".join(full_ocr).strip()
