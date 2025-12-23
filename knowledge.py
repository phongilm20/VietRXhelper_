import json
import os
import difflib

DB_FILE = "fda_database.json"

def load_database():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

DRUG_DB = load_database()

def search_fda(text_input):
    """
    Performs a fuzzy search on the local FDA database.
    Returns: Formatted string of drug details or None.
    """
    if not DRUG_DB:
        return "Error: Database file not found. Please run mining.py first."

    text_input = text_input.lower().strip()
    best_match = None
    highest_score = 0
    
    for drug in DRUG_DB:
        # Calculate string similarity ratio
        score = difflib.SequenceMatcher(None, drug['id'], text_input).ratio()
        
        # Boost score if exact match found
        if drug['id'] in text_input: 
            score = 1.0
            
        if score > 0.85 and score > highest_score:
            highest_score = score
            best_match = drug

    if best_match:
        print(f"[RAG SYSTEM] Found match in FDA DB: {best_match['brand_name']}")
        # Return formatted context for the AI
        return f"""
        Brand Name: {best_match['brand_name']}
        Active Ingredient: {best_match['generic_name']}
        Pharmacological Class (English): {best_match['pharm_class']}
        Data Source: FDA USA
        """
    
    return "Drug not found in FDA database."
