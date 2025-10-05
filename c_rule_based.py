# Define directories (update as needed)
TOKEN_DIR = "project_data/tokens"
CLEANED_DIR = "project_data/cleaned_images"
EXTRACTED_REPORTS_DIR = "project_data/extracted_reports"

# Import or define any additional functions used in the pipeline

# Module 3: Rule-Based Extraction (fast baseline) 
def group_tokens_into_lines(tokens: list, y_tolerance: int = 10) -> list:
    """Groups OCR tokens into lines based on their vertical position."""
    if not tokens:
        return []
    
    sorted_tokens = sorted(tokens, key=lambda t: (t['bbox']['top'], t['bbox']['left']))
    
    lines = []
    current_line = [sorted_tokens[0]] if sorted_tokens else []
    
    for token in sorted_tokens[1:]:
        avg_top_current_line = sum(t['bbox']['top'] for t in current_line) / len(current_line)
        
        if abs(token['bbox']['top'] - avg_top_current_line) < y_tolerance:
            current_line.append(token)
        else:
            lines.append(sorted(current_line, key=lambda t: t['bbox']['left']))
            current_line = [token]
            
    if current_line:
        lines.append(sorted(current_line, key=lambda t: t['bbox']['left']))
        
    return lines
## Patient details
def extract_patient_info(lines):
    """
    Dynamically extracts key-value pairs from header lines,
    correctly handling multiple key-value pairs and complex separators (e.g., ':-', '.:') on the same line.
    """
    data = {}
    
    key_value_pattern = re.compile(
        r"([\w\s\.'_]+?)\s*[-.\s]*[:=]\s*(.*?)(?=\s{2,}[\w\s\.'_]+?[-.\s]*[:=]|\s*$)", 
        re.IGNORECASE
    )

    for line in lines:
        if not line:
            continue
         # Reconstruct the line text, preserving original spacing
        line = sorted(line, key=lambda t: t['bbox']['left'])
        line_text = line[0]['text']
        avg_char_width = 10 # A reasonable estimate for average character width in pixels

        for i in range(len(line) - 1):
            current_token = line[i]
            next_token = line[i+1]
            
            current_right_edge = current_token['bbox']['left'] + current_token['bbox']['width']
            next_left_edge = next_token['bbox']['left']
            
            # Calculate the visual gap in pixels
            gap = next_left_edge - current_right_edge
            
            # Add spaces proportional to the visual gap
            # A large gap (e.g., > 30 pixels) will get multiple spaces
            num_spaces = 1 + int(gap / avg_char_width) if gap > (avg_char_width * 1.5) else 1
            
            line_text += (' ' * num_spaces) + next_token['text']
        
        # Pre-process the line for cleaner matching
        line_text = re.sub(r'(_)(?=\s*[-.\s]*[:=])', '', line_text) # Removes trailing underscores from keys
        line_text = line_text.replace('’', "'") # Standardizes apostrophes

        # Find all key-value pairs in the line
        matches = key_value_pattern.findall(line_text)
        
        for key, value in matches:
            clean_key = key.strip()
            clean_value = value.strip()
            
            # Add to dictionary if the key and value are valid
            if clean_key and clean_value and len(clean_key) < 35:
                data[clean_key] = clean_value
                
    return data

## Report Details
import re
import json
import numpy as np
from PIL import Image
import os
import cv2
import pytesseract
import fitz
from a_preprocessing import preprocess_image, convert_pdf_to_images



def find_table_structure(lines):
    """
    Identifies the header row and determines column boundaries by grouping header text.
    """
    header_keywords = ['test', 'investigation', 'result', 'value', 'unit', 'range']
    header_line_index = -1
    max_score = -1

    # Find the best candidate line for the header
    for i, line in enumerate(lines):
        line_text = " ".join(token['text'] for token in line).lower()
        score = 0
        if not any(char.isdigit() for char in line_text):
            score += sum(1 for keyword in header_keywords if keyword in line_text)
        if score > max_score:
            max_score = score
            header_line_index = i

    if header_line_index == -1 or max_score < 2:
        return None, None # Not a confident header

    header_line = lines[header_line_index]
    
    # Calculate average space width to distinguish between words in a header vs. new columns
    avg_char_width = np.mean([t['bbox']['width'] / len(t['text']) for t in header_line if t['text']])
    column_break_threshold = avg_char_width * 4 # Adjust if columns are too close/far

    grouped_headers = []
    current_group = []
    if header_line:
        current_group.append(header_line[0])
        for i in range(len(header_line) - 1):
            prev_token_end = header_line[i]['bbox']['left'] + header_line[i]['bbox']['width']
            next_token_start = header_line[i+1]['bbox']['left']
            
            if (next_token_start - prev_token_end) < column_break_threshold:
                current_group.append(header_line[i+1])
            else:
                grouped_headers.append(current_group)
                current_group = [header_line[i+1]]
        grouped_headers.append(current_group)

    columns = []
    for i, group in enumerate(grouped_headers):
        full_name = " ".join([t['text'] for t in group])
        start_x = group[0]['bbox']['left']
        # The end boundary is halfway to the next column's start
        if i + 1 < len(grouped_headers):
            end_x = (group[-1]['bbox']['left'] + group[-1]['bbox']['width'] + grouped_headers[i+1][0]['bbox']['left']) // 2
        else:
            end_x = 9999 # Far right for the last column
        columns.append({"name": full_name, "start_x": start_x, "end_x": end_x})
        
    return header_line_index, columns

def extract_table_rows(lines, start_index, columns, all_test_results):
    """Extracts rows from a table given the column structure."""
    current_test_name = ""
    name_col_header = columns[0]['name']

    value_col_header = None
    for col in columns:
        if 'observed' in col['name'].lower() or 'value' in col['name'].lower() or 'result' in col['name'].lower() or 'results' in col['name'].lower():
            value_col_header = col['name']
            break

    for i in range(start_index, len(lines)):
        line_tokens = lines[i]
        line_text = " ".join([t['text'] for t in line_tokens])
        line_text = line_text.replace('–', '-').replace('—', '-').replace('-',' - ')
        # Heuristic to detect end of table (e.g., summary notes, page footers)
        skip_patterns = [
            "end of report", "verified by", "page", "of", "contact", "customer care",
            "certain tests", "patient/client", "additional cost", "*teitz",
            "summary end", "note:", "printed on", "dr.", "consultant", "approved by", 
            "interpretations"
        ]
        if any(pattern in line_text for pattern in skip_patterns):
            continue
        
        if re.search(r'^\s*\w+[:/]\s*', line_text):
            continue

        row_data = {col['name']: {"value": "", "confidence": 100.0} for col in columns}
        value_present = False
        has_test_name = False
        
        for token in line_tokens:
            token_center_x = token['bbox']['left'] + token['bbox']['width'] / 2
            confidence = token.get('conf', 100) if isinstance(token.get('conf'), (int, float)) else 100
            for col in columns:
                if col['start_x'] <= token_center_x < col['end_x']:
                    row_data[col['name']]["value"] += token['text'] + " "
                    row_data[col['name']]["confidence"] = min(row_data[col['name']]["confidence"], confidence)
                    # Check if this token is a numerical value
                    if re.search(r'\d', token['text']):
                         value_present = True
                     # Track if we have a test name
                    if col['name'] == name_col_header and token['text'].strip():
                        has_test_name = True
                    break
        
         # Trim whitespace from all collected values in the row
        for key in row_data:
            row_data[key]["value"] = row_data[key]["value"].strip()
         # Get the value in the observed value column
        observed_value = row_data.get(value_col_header, {}).get("value", "") if value_col_header else ""
        
        is_valid_result = False
        
        if observed_value:
            # Check if it's a standalone numeric value (the actual test result)
            if re.match(r'([<>]|[A-Z]{1})\s*\d+\.?\d*|\d+\.?\d*$', observed_value.strip()):
            # if re.search(r'\d', observed_value) or any(c in observed_value for c in ['<', '>', ':', '/']):
                is_valid_result = True

        # Logic to handle multi-line test names
        if row_data[name_col_header]["value"] and not value_present:
            # This is likely part of a test name, not a result row
            if current_test_name:
                 current_test_name += " " + row_data[name_col_header]["value"]
            else:
                current_test_name = row_data[name_col_header]["value"]
        elif value_present and has_test_name and is_valid_result: 
            # This is a data row
            if row_data[name_col_header]["value"]:
                # If a name is present on this line, use it and clear any carry-over
                current_test_name = row_data[name_col_header]["value"]
            if not current_test_name or ':' in current_test_name:
                continue # Skip rows without a test name
            row_data[name_col_header]["value"] = current_test_name
            all_test_results.append(row_data)
            current_test_name = "" # Reset after use

def find_table_header_and_boundaries_dynamic(lines, image_cv):
    """
    Identifies the header row, intelligently groups multi-word headers, and finds column boundaries.
    """
    header_keywords = ['test', 'investigation', 'results', 'observed', 'value', 'unit', 'range', 'ref']
    line_scores = []

    for i, line in enumerate(lines):
        score = 0
        line_text = " ".join([token['text'] for token in line]).lower()
        if not line_text.strip():
            line_scores.append(-100)
            continue
        
        keyword_count = sum(1 for keyword in header_keywords if keyword in line_text)
        if keyword_count >= 2:
            score += keyword_count * 10
        if any(char.isdigit() for char in line_text):
            score -= 20
        line_scores.append(score)

    if not line_scores or max(line_scores) < 10:
        return None, None
        
    header_line_index = np.argmax(line_scores)
    header_line = lines[header_line_index]
    
    column_break_threshold = 75 

    grouped_headers = []
    current_group = []
    if header_line:
        current_group.append(header_line[0])
        for i in range(len(header_line) - 1):
            prev_token = header_line[i]
            next_token = header_line[i+1]
            gap = next_token['bbox']['left'] - (prev_token['bbox']['left'] + prev_token['bbox']['width'])
            
            if gap < column_break_threshold:
                current_group.append(next_token)
            else:
                grouped_headers.append(current_group)
                current_group = [next_token]
        grouped_headers.append(current_group)

   
    # Define column boundaries from the grouped headers
    columns = []
    for i, group in enumerate(grouped_headers):
        full_name = " ".join([t['text'] for t in group])
        start_x = group[0]['bbox']['left']
        end_x = group[-1]['bbox']['left'] + group[-1]['bbox']['width']
        
        # Extend the boundary to the start of the next column for better token capture
        if i + 1 < len(grouped_headers):
            next_start_x = grouped_headers[i+1][0]['bbox']['left']
            end_x = (end_x + next_start_x) / 2 # Midpoint between columns
            # end_x = grouped_headers[i+1][0]['bbox']['left'] -1
        else:
            end_x = image_cv.shape[1] # Extend to the edge for the last column
            
        columns.append({
            "name": full_name,
            "start_x": start_x,
            "end_x": end_x
        })
        
    return header_line_index, columns

## Process All Pages
def process_all_pages(tokens_by_page, images_by_page):
    """Orchestrates the extraction process across all pages of a report."""
    
    # Extract Patient Info
    page_1_lines = group_tokens_into_lines(tokens_by_page.get(1, []))
    patient_info = extract_patient_info(page_1_lines[:25]) # Search top part of page 1

    # Extract Table Data
    all_test_results = []
    table_columns = None
    first_data_page = 0
    for page_num in sorted(images_by_page.keys()):
        lines_on_page = group_tokens_into_lines(tokens_by_page.get(page_num, []))
        h_idx, columns = find_table_header_and_boundaries_dynamic(lines_on_page, images_by_page[page_num])
        
        if columns and not table_columns:
            table_columns = columns
            start_idx = h_idx + 1
            extract_table_rows(lines_on_page, start_idx, table_columns, all_test_results)
            first_data_page = page_num
            break
            
    if table_columns:
        for page_num in sorted(images_by_page.keys()):
            if page_num <= first_data_page: continue
            lines_on_page = group_tokens_into_lines(tokens_by_page.get(page_num, []))
            extract_table_rows(lines_on_page, 0, table_columns, all_test_results)

    # Assemble final JSON
    final_json = {
        "patient_info": patient_info,
        "lab_results": all_test_results,
        "metadata": {
            "extraction_method": "dynamic_rule_based_v6",
            "num_tests_extracted": len(all_test_results),
            "detected_columns": [col['name'] for col in table_columns] if table_columns else "Not Found"
        }
    }
    print(final_json)
    return final_json


def run_rule_based_pipeline(file_path: str):
    """
    Runs the full extraction pipeline (Modules 1-3) for a given file path.
    Returns the final extracted JSON data.
    """

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    final_json = {}
    # Check if cached tokens and images exist to skip processing
    try:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        doc.close()

        expected_token_paths = [os.path.join(TOKEN_DIR, f"{base_name}_page_{i+1:02d}_tokens.json") for i in range(num_pages)]
        expected_image_paths = [os.path.join(CLEANED_DIR, f"{base_name}_page_{i+1:02d}.png") for i in range(num_pages)]

        if all(os.path.exists(p) for p in expected_token_paths) and all(os.path.exists(p) for p in expected_image_paths):
            print(f"Found all cached files for {base_name}. Loading directly.")
            tokens_by_page = {}
            images_by_page = {}
            for i in range(num_pages):
                page_num = i + 1
                with open(expected_token_paths[i], 'r', encoding='utf-8') as f:
                    tokens_by_page[page_num] = json.load(f)
                images_by_page[page_num] = cv2.imread(expected_image_paths[i])

            if not tokens_by_page:
                return None,{"error": "No tokens were extracted from the document."}
            final_json = process_all_pages(tokens_by_page, images_by_page)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_name}_extracted_dynamic.json"
            output_path = os.path.join(EXTRACTED_REPORTS_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=4, ensure_ascii=False)
            print(f"\nSaved structured data to: {output_path}")

            return [token for page_tokens in tokens_by_page.values() for token in page_tokens], final_json

    except Exception as e:
        print(f"Could not check cache, proceeding with full processing. Error: {e}")

    # --- Module 1 & 2: Preprocessing and OCR ---
    file_extension = os.path.splitext(file_path)[1].lower()
    images_to_process = []
    if file_extension == '.pdf':
        images_to_process = convert_pdf_to_images(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        images_to_process.append(cv2.imread(file_path))

    if not images_to_process:
        return  None, {"error": f"Could not read or convert file: {file_path}"}
        
    tokens_by_page = {}
    images_by_page = {}
    all_tokens = []

    for i, img_cv in enumerate(images_to_process):
        page_num = i + 1
        cleaned_img = preprocess_image(img_cv)
        images_by_page[page_num] = cleaned_img
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_image_filename = f"{base_name}_page_{page_num:02d}.png"
        cleaned_image_path = os.path.join(CLEANED_DIR, cleaned_image_filename)
        cv2.imwrite(cleaned_image_path, cleaned_img)

        token_data = pytesseract.image_to_data(Image.fromarray(cleaned_img), output_type=pytesseract.Output.DICT)
        
        page_tokens = []
        for j in range(len(token_data['level'])):
            if int(token_data['conf'][j]) > -1 and token_data['text'][j].strip():
                page_tokens.append({
                    "text": token_data['text'][j], "conf": float(token_data['conf'][j]),
                    "bbox": {"left": token_data['left'][j], "top": token_data['top'][j], "width": token_data['width'][j], "height": token_data['height'][j]},
                    "page": page_num
                })
        tokens_by_page[page_num] = page_tokens

        token_filename = f"{base_name}_page_{page_num:02d}_tokens.json"
        token_output_path = os.path.join(TOKEN_DIR, token_filename)
        with open(token_output_path, 'w', encoding='utf-8') as f:
            json.dump(page_tokens, f, indent=4)

        all_tokens.extend(page_tokens)

    # --- Module 3: Rule-Based Information Extraction ---
    if not tokens_by_page:
        return None, {"error": "No tokens were extracted from the document."}
        
    final_json = process_all_pages(tokens_by_page, images_by_page)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_extracted_dynamic.json"
    output_path = os.path.join(EXTRACTED_REPORTS_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)
    print(f"\nSaved structured data to: {output_path}")
   
    
    return all_tokens, final_json



#Test Module 1 to 3 Pipeline
# if __name__ == "__main__":
#     # --- Execute the test on the SATHYAVATHY report ---
#     # Make sure the file is in your INPUT_DIR or provide the full path.
#     test_extraction_pipeline('project_data/raw_input/32187653_MRS. SATHYAVATHY.pdf')
#     print("\n" + "="*80 + "\n")
#     # test_extraction_pipeline_dynamic('project_data/raw_input/97308261_Mr. MANISH BAJPIE.pdf')


