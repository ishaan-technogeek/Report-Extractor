import os
import cv2
import pytesseract
import json
TOKEN_DIR = "project_data/tokens"
# Module 2: OCR & Tokenisation 
def merge_close_tokens(tokens: list, max_gap: int = 20) -> list:
    """
    Merges adjacent tokens on the same horizontal line if the gap between them is small.
    This fixes fragmented names and compound words (e.g., "John" + "Doe" -> "John Doe").
    """
    if not tokens:
        return []
    
    merged_tokens = []
    current_phrase = tokens[0]
    
    for i in range(1, len(tokens)):
        next_token = tokens[i]
        
        # Calculate horizontal distance between tokens
        current_right = current_phrase['bbox']['left'] + current_phrase['bbox']['width']
        gap = next_token['bbox']['left'] - current_right
        
        # Check vertical alignment (y-overlap) and horizontal proximity
        y_overlap = abs(current_phrase['bbox']['top'] - next_token['bbox']['top'])
        
        # If the gap is small and they are vertically aligned (small y_overlap)
        if gap < max_gap and y_overlap < 5:
            # Merge: update text, width, and confidence (use min confidence)
            current_phrase['text'] += ' ' + next_token['text']
            current_phrase['bbox']['width'] += next_token['bbox']['width'] + gap
            current_phrase['conf'] = min(current_phrase['conf'], next_token['conf'])
        else:
            # Not mergeable, finalize the current phrase and start a new one
            merged_tokens.append(current_phrase)
            current_phrase = next_token
            
    merged_tokens.append(current_phrase)
    return merged_tokens

def run_ocr_and_tokenize(image_path: str, report_name: str) -> str:
    """Runs OCR on a cleaned image and saves word-level tokens with bboxes."""
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # Use image_to_data to get word-level details (text, box, confidence)
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        tokens_list = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # level 5 is word level
            if data['level'][i] == 5 and data['text'][i].strip():
                token = {
                    "text": data['text'][i],
                    "conf": float(data['conf'][i]) / 100, # Convert confidence to 0.0 to 1.0
                    "bbox": {
                        "left": data['left'][i],
                        "top": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                }
                tokens_list.append(token)

        merged_tokens = merge_close_tokens(tokens_list)

        # Save tokens to a JSON file
        output_filename = f"{report_name}_tokens.json"
        output_path = os.path.join(TOKEN_DIR, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(merged_tokens, f, indent=4)
            
        print(f"Saved {len(merged_tokens)} tokens to: {output_path}")
        return output_path

    except Exception as e:
        print(f" Module 2 Failed during OCR: {e}")
        return None

# ## Test Module 2
# if 'cleaned_files' in locals() and cleaned_files:
#     token_file_path = run_ocr_and_tokenize(cleaned_files[0], base_name)
    
#     if token_file_path:
#         with open(token_file_path, 'r') as f:
#             tokens = json.load(f)
            
#         print("\n Module 2 Success. First 10 tokens extracted:")
#         for token in tokens[:10]:
#             print(f" - Text: '{token['text']}', Conf: {token['conf']:.2f}, BBox: {token['bbox']}")
    
# else:
#     print(" Skipping Module 2 Test: No cleaned image found from Module 1.")

