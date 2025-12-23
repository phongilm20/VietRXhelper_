<p align="center">
  <img src="logo_VietRXhelper.png" alt="VietRx Helper Logo" height="300">
</p>

# VietRx Helper: AI-Powered Medical Assistant for the Elderly

> **A safety-first, multimodal AI system designed to bridge the health literacy gap for Vietnamese immigrants using Computer Vision, RAG, and a Dual-LLM Audit Architecture.**

---

## 📖 Overview

**VietRx Helper** is an intelligent assistive technology designed to help elderly Vietnamese individuals understand complex medication labels. Unlike standard translation tools, this system prioritizes **medical safety** and **cultural adaptation**.

The core innovation lies in its **Dual-LLM Safety Architecture**, where one AI generates empathetic advice while a second "Auditor" AI verifies the content against official FDA records to prevent hallucinations before the user hears a single word.

## 🚀 Key Features

* **👁️ End-to-End Computer Vision:**

  * Utilizes **YOLOv8 (Nano)** fine-tuned on a custom drug dataset to detect drug names on bottle labels.
  * Integrates **EasyOCR** with grayscale + threshold preprocessing for robust text extraction.

* **🛡️ Dual-LLM Safety Protocol (Reviewer-Refiner):**

  * **The Doctor (Generator):** Drafts culturally appropriate advice using polite Vietnamese honorifics.
  * **The Auditor (Evaluator):** Performs strict fact-checking using cross-referenced FDA data.

* **📚 RAG (Retrieval-Augmented Generation):**

  * Anchors generation to a verified local FDA Knowledge Base built from the OpenFDA API.

* **🧠 Cultural Adaptation (Generative AI):**

  * Uses **Google Gemini 2.5 Flash** to translate pharmacological terminology into easy-to-understand Vietnamese.

* **🗣️ Text-to-Speech (TTS):**

  * Outputs validated advice via natural Vietnamese speech.

## 🏗️ System Architecture

1. **Data Acquisition Layer:** ETL script (`mining.py`) pulls and cleans NDC/FDA drug data.

2. **Vision Layer:**

   * Raw Image → YOLOv8 Detection → Cropping → OCR.
   * Output: Drug name candidate.

3. **Knowledge Retrieval Layer:**

   * Fuzzy string matching → Retrieve Ground Truth (ingredients, pharm class).

4. **Dual-LLM Reasoning Layer:**

   * **Doctor Agent:** draft response.
   * **Auditor Agent:** validates accuracy & forces regeneration if mismatched.

5. **Interaction Layer:**

   * Sanitizes text → TTS audio output.

## 📂 Project Structure

```text
VietRx-Project/
├── models/
│   └── best.pt               
├── src/                      
│   ├── main.py               
│   ├── vision.py
│   ├── knowledge.py
│   └── brain.py              
├── tests_webcam/             
│   ├── main_test.py          
│   ├── vision_test.py        
│   └── knowledge_test.py     
├── fda_database.json         
├── requirements.txt          
├── .env
└── README.md                 
```

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.9+
* Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/phongilm20/VietRx-Helper.git
cd VietRx-Helper
```

### 2. Install Dependencies

```bash
pip install requests google-genai ultralytics easyocr opencv-python gTTS
```

### 3. Setup Environment

Replace the placeholder API Key in `brain.py` with your actual Google Gemini API Key.

### 4. Initialize Database

```bash
python mining.py
```

*(Generates ~5000 FDA-verified drug records.)*

## 💻 Usage

1. Add an image of a medication bottle (e.g., `test.jpg`).
2. Run:

```bash
python main.py
```

3. Pipeline:

   * Detect drug name.
   * Retrieve FDA data.
   * Doctor AI drafts advice.
   * Auditor AI validates correctness.
   * Audio is generated and played.

## 🔬 Tech Stack

* **Language:** Python 3.10
* **Computer Vision:** YOLOv8, EasyOCR, OpenCV
* **Generative AI:** Google Gemini 2.5 Flash (Dual-Layer Architecture)
* **Database:** OpenFDA
* **Audio:** gTTS

## ⚠️ Disclaimer

This is a research prototype for educational purposes and **not** a medical substitute.

## 👨‍💻 Author

**Nguyen Phong**

* Department of Computer Science & Engineering
* Wright State University
