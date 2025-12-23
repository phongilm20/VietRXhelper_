import vision
import knowledge
import brain
import os
import platform
import re
import difflib
from gtts import gTTS

def clean_text_for_audio(text):
    """
    Utility: Sanitizes the generated text for optimal Speech Synthesis.
    Removes Markdown artifacts and normalizes whitespace.
    """
    text = re.sub(r'[*#_`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def play_audio(text):
    """
    Executes cross-platform audio playback using OS-level commands.
    Handlers: 'start' (Windows), 'afplay' (macOS), 'xdg-open' (Linux).
    """
    print("[INFO] Synthesizing speech audio (TTS)...")
    try:
        clean_response = clean_text_for_audio(text)
        # Google TTS (gTTS) API call
        tts = gTTS(text=clean_response, lang='vi')
        filename = "advice.mp3"
        tts.save(filename)
        
        # Execute system command for playback
        if platform.system() == "Windows":
            os.system(f'start {filename}')
        elif platform.system() == "Darwin": 
            os.system(f'afplay {filename}')
        else: # Linux
            os.system(f'xdg-open {filename}')
    except Exception as e:
        print(f"[ERROR] Audio playback failed: {e}")

def run_system():
    # --- SYSTEM INITIALIZATION ---
    print("Initializing VietRx System v1.0...")
    print("[INFO] Loading dependency modules...")
    
    image_path = "test.jpg" 
    
    if not os.path.exists(image_path):
        print(f"[CRITICAL] Input file not found: {image_path}")
        return

    # -------------------------------------------------------------------------
    # PHASE 1: COMPUTER VISION PIPELINE (YOLOv8 + EasyOCR)
    # -------------------------------------------------------------------------
    print(f"[INFO] Processing input image: {image_path}")
    
    # Analyze image to extract raw unstructured text
    raw_ocr_text = vision.analyze_image(image_path)
    
    # -------------------------------------------------------------------------
    # PHASE 2: POST-OCR ERROR CORRECTION (ENTITY LINKING)
    # -------------------------------------------------------------------------
    # Algorithm: Fuzzy String Matching (Levenshtein Distance)
    # Objective: Map noisy OCR output to the nearest valid entity in Ground Truth.
    # -------------------------------------------------------------------------

    database_ref = knowledge.DRUG_DB  # Reference to in-memory database
    best_candidate = raw_ocr_text     # Default fallback
    highest_confidence_score = 0.0
    CONFIDENCE_THRESHOLD = 0.4        # Minimum similarity ratio for acceptance

    # Heuristic: Only attempt correction if OCR signal is sufficient (>3 chars)
    if len(raw_ocr_text) > 3 and database_ref:
        for record in database_ref:
            target_name = record['brand_name']
            
            # Compute similarity ratio [0.0 - 1.0]
            similarity_score = difflib.SequenceMatcher(None, target_name.lower(), raw_ocr_text.lower()).ratio()
            
            # Boost Heuristic: Assign high confidence if target is a substring
            # This handles cases like: OCR="100mg Bexarotene Tabs" -> Target="Bexarotene"
            if target_name.lower() in raw_ocr_text.lower():
                similarity_score = 0.95 
            
            # Greedy Selection: Keep the candidate with the highest global score
            if similarity_score > highest_confidence_score:
                highest_confidence_score = similarity_score
                best_candidate = target_name

    # Determine final suggestion based on threshold logic
    final_suggestion = best_candidate if highest_confidence_score > CONFIDENCE_THRESHOLD else raw_ocr_text
    
    # Log raw data for debugging/audit
    print(f"[OCR RAW] Signal: '{raw_ocr_text}'")
    print(f"[ENTITY LINKING] Candidate: '{final_suggestion}' | Score: {highest_confidence_score:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 3: HUMAN-IN-THE-LOOP VERIFICATION
    # -------------------------------------------------------------------------
    # Safety Protocol: User must confirm the detected entity before medical lookup.
    confirm = input(f"[INPUT] Confirm drug name '{final_suggestion}'? (Y/n): ").strip()
    
    if confirm.lower() in ("y", "yes", ""):
        drug_name = final_suggestion
    else:
        drug_name = confirm
        print(f"[INFO] Manual override by user: '{drug_name}'")

    # -------------------------------------------------------------------------
    # PHASE 4: RETRIEVAL-AUGMENTED GENERATION (RAG)
    # -------------------------------------------------------------------------
    print(f"[RAG] Querying FDA Knowledge Base for: '{drug_name}'")
    fda_info = knowledge.search_fda(drug_name)
    
    # -------------------------------------------------------------------------
    # PHASE 5: DUAL-LLM REASONING & AUDIT
    # -------------------------------------------------------------------------
    print("[LLM] Generating medical advice with Auditor validation...")
    advice = brain.get_medical_advice(drug_name, fda_info)
    
    # -------------------------------------------------------------------------
    # PHASE 6: OUTPUT & ACCESSIBILITY
    # -------------------------------------------------------------------------
    final_output = clean_text_for_audio(advice)
    
    print("\n[OUTPUT MESSAGE]")
    print(final_output)
    print("") # End of stream
    
    play_audio(final_output)

if __name__ == "__main__":
    run_system()

