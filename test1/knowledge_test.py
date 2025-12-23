"""
VietRx Logic Module: Performs entity linking and medical metadata extraction.
"""

import knowledge  # Access original drug database
import difflib
import re

def analyze_metadata(detections):
    """
    Standardizes raw OCR output into structured medical entities.
    Algorithm: Fuzzy String Matching (Levenshtein Distance) + Regex Extraction.
    """
    database_ref = knowledge.DRUG_DB
    best_candidate = "Unknown"
    highest_score = 0.0
    
    # Entity Storage
    strength = "N/A"
    quantity = "N/A"
    expiry = "N/A"

    for d in detections:
        text = d['text']
        
        # 1. Fuzzy Entity Linking: Match OCR to Ground Truth
        for record in database_ref:
            target_name = record['brand_name']
            # Calculate similarity ratio [0.0 - 1.0]
            score = difflib.SequenceMatcher(None, target_name.lower(), text.lower()).ratio()
            
            # Boost Heuristic: Substring membership significantly increases score
            if target_name.lower() in text.lower():
                score = 0.95 
            
            if score > highest_score:
                highest_score = score
                best_candidate = target_name

        # 2. Heuristic Extraction: Targeted Regex for medical units
        # Dosage Strength (mg, ml, mcg, g)
        s_match = re.search(r'(\d+)\s*(mg|ml|mcg|g)', text, re.I)
        if s_match: strength = s_match.group(0)
        
        # Packaging Quantity (tablets, capsules, etc.)
        q_match = re.search(r'(\d+)\s*(capsules|tablets|pills|vien)', text, re.I)
        if q_match: quantity = q_match.group(0)

        # Expiry Date Detection (EXP, HSD formats)
        e_match = re.search(r'(EXP|HSD|Expiry)[\s:]*(\d+/\d+)', text, re.I)
        if e_match: expiry = e_match.group(0)

    return {
        "final_suggestion": best_candidate,
        "score": highest_score,
        "strength": strength,
        "quantity": quantity,
        "expiry": expiry,
        "fda_record": knowledge.search_fda(best_candidate) if highest_score > 0.4 else None
    }