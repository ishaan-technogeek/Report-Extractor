# Imports & Initialisation

# ! pip install -qq PyMuPDF opencv-python numpy pandas Pillow pytesseract spacy scikit-learn
# ! pip install -qq fastapi uvicorn python-multipart

import os
import json
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import pytesseract
import spacy

from c_rule_based import run_rule_based_pipeline
# DATA_DIR, INPUT_DIR, CLEANED_DIR, TOKEN_DIR, EXTRACTED_REPORTS_DIR, CORRECTIONS_DIR, DUMMY_PDF_PATH = (None,) * 7

DATA_DIR = 'project_data'

# Define subdirectories for inputs, outputs, and training data
INPUT_DIR = os.path.join(DATA_DIR, 'raw_input')
CLEANED_DIR = os.path.join(DATA_DIR, 'processed_images')
TOKEN_DIR = os.path.join(DATA_DIR, 'tokens')
EXTRACTED_REPORTS_DIR = os.path.join(DATA_DIR, 'extracted_reports')
CORRECTIONS_DIR = os.path.join(DATA_DIR, 'training_data_corrections')
PENDING_REPORTS_DIR = os.path.join(DATA_DIR, 'pending_reports')
# Create directories if they do not exist
for d in [INPUT_DIR, CLEANED_DIR, TOKEN_DIR, EXTRACTED_REPORTS_DIR, CORRECTIONS_DIR, PENDING_REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


try:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract' 
except:
    print("WARNING: Tesseract command not found. Please set the path correctly.")


# 3. Create a dummy input file for testing 
# DUMMY_PDF_PATH = os.path.join(INPUT_DIR, '17756177_50641000301.pdf')
DUMMY_PDF_PATH = os.path.join(INPUT_DIR, '32187653_MRS. SATHYAVATHY.pdf')
if not os.path.exists(DUMMY_PDF_PATH):
    print(f"Placeholder: Create a dummy PDF at {DUMMY_PDF_PATH} for testing.")
    # In a real scenario, you would manually place a report PDF here.
    # For now, we will assume one exists for the code structure demonstration.
    
print("Setup complete. Directories created and paths defined.")



# Module 6 : Load trained spaCy model

MODEL_PATH = "./models/model-best"
nlp_model = None
if os.path.exists(MODEL_PATH):
    nlp_model = spacy.load(MODEL_PATH)
    print(" Trained spaCy model loaded successfully from models/model-best.")
else:
    print(" WARNING: Trained spaCy model not found at models/model-best. Inference will use rule-based fallback only.")


def process_model_entities(doc, tokens):
    """
    Converts spaCy's entity predictions into a structured JSON format.
    Groups lab results by their vertical position (y-coordinate).
    """
    patient_info = {}
    lab_results_by_line = {} # Use y-coord as a key to group entities
    y_tolerance = 15 # Pixels

    # Create a mapping from character index to token for bbox lookup
    char_to_token_idx = {}
    current_char = 0
    for i, token in enumerate(tokens):
        for _ in range(len(token['text'])):
            char_to_token_idx[current_char] = i
            current_char += 1
        current_char += 1 # For the space

    for ent in doc.ents:
        # Find the approximate y-coordinate of the entity
        start_token_idx = char_to_token_idx.get(ent.start_char)
        if start_token_idx is None:
            continue
        
        y_coord = tokens[start_token_idx]['bbox']['top']

        # Group patient info directly
        if ent.label_ in ['Patient Name', 'Age', 'Gender', 'Patient ID', 'PATIENT_NAME', 'AGE', 'GENDER']:
            patient_info[ent.label_.replace(' ', '_')] = ent.text
        else:
            # Group lab results by vertical line
            found_line = False
            for y_key in lab_results_by_line:
                if abs(y_key - y_coord) < y_tolerance:
                    lab_results_by_line[y_key][ent.label_] = ent.text
                    found_line = True
                    break
            if not found_line:
                lab_results_by_line[y_coord] = {ent.label_: ent.text}

    # Convert the grouped lab results into a list of dicts
    lab_results = list(lab_results_by_line.values())

    return {"patient_info": patient_info, "lab_results": lab_results}



def run_inference_pipeline(file_path: str):
    """
    Runs the full inference pipeline.
    1. Tries ML model extraction.
    2. Falls back to rule-based extraction if ML fails or is insufficient.
    """
    # Step 1 & 2: Preprocessing and OCR to get tokens
    # The rule-based function already does this, so we can call it to get the tokens
    all_tokens, rule_based_result = run_rule_based_pipeline(file_path)
    final_result = None
    if not all_tokens:
         return {"error": "OCR failed to extract any tokens from the document."}

    # Step 3: Try ML Model first
    if nlp_model:
        print("--- Attempting ML Model Extraction ---")
        full_text = " ".join([t['text'] for t in all_tokens])
        doc = nlp_model(full_text)
        ml_result = process_model_entities(doc, all_tokens)
        
        # Step 5: Fallback Logic
        # Check if the ML result is good enough (e.g., has patient info AND lab results)
        if ml_result.get("patient_info") and ml_result.get("lab_results"):
            print("Using ML Model for extraction.")
            ml_result["metadata"] = {"extraction_method": "ml_model"}
            final_result = ml_result
            return final_result            
        else:
            print(" ML model result was insufficient. Falling back to rule-based method.")

    final_result = rule_based_result

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_pending.json"
    output_path = os.path.join(PENDING_REPORTS_DIR, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4)
        print(f" Saved pending JSON to: {output_path}")
    except Exception as e:
        print(f" Error saving pending JSON: {e}")
    
    # If ML model is not available or failed, return the rule-based result
    print("Using Rule-Based Fallback for extraction.")
    return final_result
