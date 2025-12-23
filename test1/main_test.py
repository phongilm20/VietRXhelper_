"""
VietRx Controller: Orchestrates UI, Vision, Logic, and LLM reasoning.
"""

import sys
import os
import cv2
import re  # FIXED: Required for clean_text_for_audio
import platform
from gtts import gTTS

# Configure system path to allow module cross-referencing in subfolders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vision_test
import knowledge_test
import brain  # Accesses Gemini/LLM API and Safety Auditor
    
def clean_text_for_audio(text):
    """Sanitizes generated LLM text for optimal TTS playback."""
    text = re.sub(r'[*#_`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def play_audio(text):
    """Executes cross-platform audio synthesis and playback."""
    try:
        clean_response = clean_text_for_audio(text)
        tts = gTTS(text=clean_response, lang='vi')
        filename = "advice.mp3"
        tts.save(filename)
        
        # Platform-specific execution commands
        if platform.system() == "Windows": os.system(f'start {filename}')
        else: os.system(f'xdg-open {filename}')
    except Exception as e:
        print(f"[ERROR] Audio Service Failure: {e}")

def run_system():
    """Main execution loop for real-time webcam analysis."""
    vs = vision_test.VisionSystem() #
    cap = cv2.VideoCapture(0)
    
    print("--- VietRx Academic Prototype Ready ---")
    print("Command: Press 's' to Scan Medication, 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("VietRx - Real-time Scanning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\n[STEP 1] Initializing Vision Pipeline...")
            detections = vs.extract_text_proposals(frame) #
            
            print("[STEP 2] Processing Metadata & Entity Linking...")
            meta = knowledge_test.analyze_metadata(detections) #
            
            # Print analysis results for debugging/audit
            print(f"[ENTITY] Candidate: '{meta['final_suggestion']}' (Conf: {meta['score']:.2f})")
            print(f"[INFO] Strength: {meta['strength']} | Quantity: {meta['quantity']} | Exp: {meta['expiry']}")

            # Step 3: Human-in-the-Loop Validation
            confirm = input(f"[INPUT] Confirm identification '{meta['final_suggestion']}'? (Y/n): ").strip()
            drug_name = meta['final_suggestion'] if confirm.lower() in ("y", "yes", "") else confirm

            # Step 4: FDA Knowledge Retrieval & LLM Advice
            fda_info = knowledge_test.knowledge.search_fda(drug_name)
            context = f"Drug: {drug_name}, Dosage: {meta['strength']}, Qty: {meta['quantity']}, Exp: {meta['expiry']}"
            
            print("[STEP 3] Executing LLM Safety Audit & Advice Generation...")
            advice = brain.get_medical_advice(context, fda_info) #
            
            print(f"\n[FINAL OUTPUT]:\n{advice}")
            play_audio(advice) #

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()