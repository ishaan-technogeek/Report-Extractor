import os
import cv2
import numpy as np
from PIL import Image

import fitz  # PyMuPDF

# Define CLEANED_DIR if not already defined elsewhere
CLEANED_DIR = "project_data/cleaned_images"
# Module 1: File Input & Preprocessing 

def deskew_image(image_cv: np.ndarray) -> tuple[np.ndarray, float]:
    """Applies a simple skew correction using image moments."""
    if len(image_cv.shape) == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv

    # Simple binary mask for contour finding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    
    if coords.size == 0:
        return image_cv, 0
        
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_cv, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle

def preprocess_image(img_cv: np.ndarray) -> np.ndarray:
    """Applies deskewing, grayscale conversion, and OTSU thresholding."""
    if img_cv is None:
        return None

    # 1. Deskew
    deskewed_img = img_cv
    # deskewed_img, angle = deskew_image(img_cv)
    
    # 2. Convert to Grayscale
    if len(deskewed_img.shape) == 3:
        gray_img = cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = deskewed_img

    # 3. Denoise and Threshold (OTSU) for optimal OCR quality
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, final_clean_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return final_clean_img

def convert_pdf_to_images(file_path: str) -> list[np.ndarray]:
    """Converts PDF pages to high-resolution OpenCV image arrays using PyMuPDF."""
    images = []
    try:
        doc = fitz.open(file_path)
        zoom = 4.16 # ~300 DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
            images.append(img_cv)
            
        doc.close()
    except Exception as e:
        print(f"Error processing PDF with PyMuPDF: {e}")
    return images

def process_report_file(file_path: str, output_base_name: str) -> list[str]:
    """Main function for Module 1: Handles input, preprocesses, and saves cleaned images."""
    cleaned_image_paths = []
    file_extension = file_path.lower().split('.')[-1]
    images_to_process = []

    if file_extension == 'pdf':
        print(f"Processing PDF: {file_path}")
        images_to_process = convert_pdf_to_images(file_path)
    
    elif file_extension in ['jpg', 'jpeg', 'png']:
        print(f"Processing image: {file_path}")
        img_cv = cv2.imread(file_path)
        if img_cv is not None:
            images_to_process.append(img_cv)
    
    else:
        print(f"Unsupported file type: {file_extension}")
        return []

    for i, img_cv in enumerate(images_to_process):
        cleaned_img_cv = preprocess_image(img_cv)
        
        output_filename = f"{output_base_name}_page_{i+1:02d}.png"
        output_path = os.path.join(CLEANED_DIR, output_filename) 
        cv2.imwrite(output_path, cleaned_img_cv)
        cleaned_image_paths.append(output_path)
        print(f"Saved cleaned image to: {output_path}")

    return cleaned_image_paths
# ## Test Module 1
# report_file_path = DUMMY_PDF_PATH
# base_name = os.path.splitext(os.path.basename(report_file_path))[0]

# if os.path.exists(report_file_path):
#     cleaned_files = process_report_file(report_file_path, base_name)
    
#     if cleaned_files:
#         print(f"\n✅ Module 1 Success. Found {len(cleaned_files)} cleaned image(s).")
#         print("Displaying the first cleaned image (Binary, Deskewed):")
        
#         # Display the image in the notebook
#         try:
#             display_img = Image.open(cleaned_files[0])
#             # display(display_img)
#         except Exception as e:
#             print(f"Could not display image: {e}")
#     else:
#         print("❌ Module 1 Failed: Check file path and dependencies (PyMuPDF, Poppler).")
# else:
#     print("⚠️ Skipping Module 1 Test: Dummy PDF not found.")