# Lab Report Digitization and Structured Data Extraction

This project is a comprehensive system designed to automatically extract and digitize patient details and lab test results from various formats of medical reports (PDFs, JPGs, PNGs) into a structured JSON format. It features a machine learning component that learns from user corrections, improving its accuracy over time.


*User interface for correcting and verifying extracted lab report data.*

---
## Objective
The primary goal is to build an intelligent pipeline that can:
* Take a lab report file (PDF or image) as input.
* Recognize and extract key information (patient details, test names, values, units, reference ranges).
* Convert the extracted data into a structured JSON format.
* Provide a user-friendly interface for verifying and correcting the data.
* Continuously learn from user-provided corrections to improve the extraction model.

---
## Key Features
* **Multi-Format Input**: Accepts lab reports in PDF, JPG, or PNG formats.
* **Image Preprocessing**: Automatically converts PDFs to images, cleans up scanned files (deskewing, denoising), and enhances them for optimal OCR performance.
* **OCR & Text Extraction**: Utilizes Tesseract for accurate text extraction from images, capturing word positions (bounding boxes) for contextual analysis.
* **Hybrid Extraction Model**: Employs a spaCy-based NER model for intelligent data extraction and falls back on a robust rule-based system for reliability.
* **Human-in-the-Loop UI**: A simple web interface allows users to review, edit, and confirm the extracted data, ensuring high accuracy.
* **Continuous Learning**: User corrections are saved and can be used to retrain and improve the machine learning model over time.
* **REST API**: A FastAPI-based backend provides endpoints to upload reports and save corrected data programmatically.
* **Performance Evaluation**: Includes a script to measure the model's performance using metrics like Precision, Recall, and F1-Score against a ground-truth dataset.

---
## How It Works
The system follows a multi-stage pipeline:

1.  **File Upload**: A user uploads a lab report via the web interface.
2.  **Preprocessing (Module 1)**: The input file is converted into cleaned, high-resolution images.
3.  **OCR & Tokenization (Module 2)**: Tesseract OCR reads the text and its coordinates from the cleaned images.
4.  **Inference (Module 6)**: The trained spaCy NER model attempts to extract entities from the tokenized text.
5.  **Fallback Mechanism**: If the ML model's output is insufficient, the system automatically falls back to a dynamic **Rule-Based Extraction (Module 3)** method to ensure data is captured.
6.  **Human Review (Module 4)**: The extracted data is displayed on a web form where a user can make corrections and save the final, accurate version.
7.  **Data Storage (Module 8)**: The corrected data is saved, creating a valuable dataset for retraining the ML model.

---
## Tech Stack
* **Backend**: Python, FastAPI 
* **OCR Engine**: Tesseract 
* **Machine Learning**: spaCy, Scikit-learn 
* **Image Processing**: OpenCV, PyMuPDF 
* **Core Libraries**: Pandas, NumPy

## Setup and Usage

### 1. Installation
Clone the repository and install the required dependencies. It's recommended to use a virtual environment.
```
# It is assumed you have Tesseract installed on your system
---
## Project Structure
.
├── app.py                      # FastAPI application with API endpoints.
├── a_preprocessing.py          # Module for image cleaning and enhancement.
├── b_ocr.py                    # Module for running Tesseract OCR on images.
├── c_rule_based.py             # Rule-based fallback extraction logic.
├── d_training-report-extraction.ipynb # Notebook for preparing training data.
├── e_inference.py              # Module for running the spaCy NER model.
├── evaluate.ipynb              # Jupyter notebook to evaluate model accuracy.
├── index.html                  # Frontend HTML for the user interface.
├── config.cfg                  # Configuration file for the spaCy training pipeline.
├── requirements.txt            # List of all Python dependencies.
│
├── models/
│   └── model-best/             # Directory for the trained spaCy model.
│
└── project_data/
    ├── raw_input/              # Stores uploaded lab reports.
    ├── processed_images/       # Stores cleaned images after preprocessing.
    ├── tokens/                 # Stores OCR output (tokens and bounding boxes).
    ├── final_reports/          # Stores final JSON output from the pipeline.
    └── training_data_corrections/ # Stores user-corrected JSON files for retraining.
```

### 2. Run the Server
Start the FastAPI application using Uvicorn.

Bash :

``` uvicorn app:app --reload ```

The application will be available at http://127.0.0.1:8000.

### 3. Using the Application
Upload: Open your web browser to http://127.0.0.1:8000. Use the UI to upload a lab report file (PDF, JPG, or PNG).

Verify & Correct: The system will process the file and display the extracted data in a form. Review the fields, make any necessary corrections, add or delete rows, and click "Save Corrected Data".

Saved: The corrected data is saved to the project_data/training_data_corrections/ directory, ready to be used for future model training.



# Evaluation
The evaluate.ipynb notebook measures the performance of the extraction pipeline. It processes a set of test files, compares the extracted results against hand-corrected "ground truth" files, and calculates precision, recall, and F1-score.
