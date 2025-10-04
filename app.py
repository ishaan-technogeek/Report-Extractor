import os
import shutil
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List


from main_processor import (run_inference_pipeline,
DATA_DIR, INPUT_DIR, CLEANED_DIR, TOKEN_DIR, EXTRACTED_REPORTS_DIR, CORRECTIONS_DIR, DUMMY_PDF_PATH)


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serves the main HTML page for the UI."""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload/")
async def upload_and_process_report(file: UploadFile = File(...)):
    """
    Accepts a PDF, runs the full extraction pipeline (Modules 1-3),
    and returns the extracted JSON data.
    """
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"--- Processing: {file.filename} ---")
    
    try:
        extracted_data = run_inference_pipeline(file_path)
        # Check if the pipeline returned an error
        if "error" in extracted_data:
            return JSONResponse(status_code=400, content=extracted_data)
        
        # Add filename to metadata for the save function
        extracted_data.setdefault('metadata', {})['original_filename'] = file.filename
        
        # Return the extracted data as JSON
        return JSONResponse(status_code=200, content=extracted_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})


@app.post("/save/")
async def save_corrected_data(corrected_data: dict):
    """
    Receives the user-corrected JSON and saves it to the corrections directory.
    This directory is specified by the CORRECTIONS_DIR global variable.
    """
    original_filename = corrected_data.get("metadata", {}).get("original_filename", "unknown_file.pdf")
    base_name = os.path.splitext(original_filename)[0]
    
    # Use the CORRECTIONS_DIR path defined at the top of the script
    output_filename = f"{base_name}_corrected.json"
    output_path = os.path.join(CORRECTIONS_DIR, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=4, ensure_ascii=False)
        print(f"Corrected data saved to: {output_path}")
        return JSONResponse(status_code=200, content={"message": f"Successfully saved corrected data to {output_path}"})
    except Exception as e:
        print(f"Error saving file: {e}")
        return JSONResponse(status_code=500, content={"error": f"Could not save the file: {str(e)}"})

if __name__ == "__main__":
    # To run this app: `uvicorn app:app --reload` in your terminal
    uvicorn.run(app, host="0.0.0.0", port=8000)