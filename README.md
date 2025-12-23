<p align="center">
  <img src="logo_VietRXhelper.png" alt="VietRx Helper Logo" height="250">
</p>

# VietRx Helper: Multimodal Assistive Technology for Pharmaceutical Label Interpretation

> **Abstract:** This project presents a safety-oriented, multimodal AI system designed to mitigate health literacy barriers for Vietnamese immigrants. The system integrates real-time Computer Vision (YOLOv8, EasyOCR), Retrieval-Augmented Generation (RAG) using FDA-validated datasets, and a Dual-LLM Audit Architecture to ensure pharmaceutical accuracy and cultural relevance.

---

## 1. Research Objectives and Overview

VietRx Helper is an assistive technology developed to provide elderly users with accurate, culturally adapted interpretations of medication labels. Unlike conventional translation tools, this system prioritizes clinical safety by anchoring Large Language Model (LLM) responses to verified medical databases. The primary goal is to provide a reliable interface that converts complex pharmacological data into simplified, audible Vietnamese guidance.

## 2. Technical Methodology: The Dual-Agent Safety Architecture

To prevent AI hallucinations in medical contexts, the system employs a "Reviewer-Refiner" paradigm:

1. **Generation (The Doctor Agent):** Utilizes Google Gemini to synthesize raw OCR data and FDA metadata into an empathetic response using Vietnamese honorifics.
2. **Validation (The Auditor Agent):** A secondary logic gate that performs a strict fact-check of the generated advice against source FDA records.
3. **Conflict Resolution:** If the Auditor detects discrepancies (e.g., fabricated dosages), the system triggers a recovery protocol to issue a safe, generalized warning instead of potentially harmful misinformation.

## 3. Implementation Details: Computer Vision and Entity Extraction

The system utilizes a custom-trained YOLOv8 model for label localization and EasyOCR for text recognition. To bridge raw OCR output with structured data, the system implements:
* **Fuzzy String Matching:** Employs Levenshtein distance to map noisy OCR candidates to verified FDA entries.
* **Heuristic Metadata Extraction:** A Regex-based engine designed to identify Dosage Strength (mg/ml), Quantity (Tablets/Capsules), and Expiry Dates.

## 4. System Demonstration Activity

### 4.1 Visual Identification Performance
The following figure demonstrates the system's ability to localize medication labels and extract critical metadata in a real-world environment via a live video stream.

![VietRx Application Demo](demo_capture.png)
*Figure 1: Real-time identification and metadata extraction of a pharmaceutical container.*

### 4.2 Audible Guidance Output
The output of the Dual-LLM pipeline is converted into natural speech. This sample demonstrates the audited, culturally adapted medical advice provided to the user:

[🔊 Listen to Sample Medical Advice (advice.mp3)](advice.mp3)

## 5. Operational Workflow and Installation

### 5.1 System Requirements
* **Environment:** Python 3.10+
* **Dependencies:** ultralytics, easyocr, opencv-python, google-genai, gTTS, pygame, python-dotenv.

### 5.2 Execution Protocol
To evaluate the real-time prototype, execute the following command:
`python main_test.py`

* **Interaction Steps:**
  - Press 's' to initiate frame capture and analysis.
  - Verify the detected drug name via the terminal prompt.
  - The system will automatically synthesize and play the audited Vietnamese audio advice.
  - Press 'q' to terminate the session.

## 6. Project Structure

VietRXhelper_/
├── best.pt               # YOLOv8 custom weights
├── brain.py              # LLM Integration & Safety Auditor
├── fda_database.json     # Local FDA Knowledge Base
├── knowledge.py          # FDA Database lookup logic
├── knowledge_test.py     # Advanced entity extraction (Webcam version)
├── main.py               # Static image processing entry point
├── main_test.py          # Real-time Webcam Controller (Main Entry)
├── mining.py             # ETL script for FDA data
├── vision.py             # OCR module for files
├── vision_test.py        # Video frame processing module
├── requirements.txt      # Dependency manifest
├── advice.mp3            # Sample audio output
├── demo_capture.png      # System demonstration image
└── README.md             # Project documentation

## 7. Author and Institutional Affiliation

Xuan Thanh Phong Nguyen
Computer Engineering Student
Department of Computer Science and Engineering
Wright State University, Dayton, Ohio

---
*Disclaimer: This system is a research prototype for academic purposes and does not constitute professional medical advice.*
