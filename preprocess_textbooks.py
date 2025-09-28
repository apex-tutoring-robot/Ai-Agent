import fitz  # PyMuPDF
import os
import json

pdf_path = 'textbooks/CA CSS Math - Content Standards (CA Dept of Education) - ccssmathstandardaug2013.pdf'
output_dir = 'processed_json_textbooks'
os.makedirs(output_dir, exist_ok=True)

grade_triggers = {
    "Kindergarten": {"id": "K", "filename": "math_kindergarten.json"},
    "Grade 1": {"id": "1", "filename": "math_grade_1.json"},
    "Grade 2": {"id": "2", "filename": "math_grade_2.json"},
    "Grade 3": {"id": "3", "filename": "math_grade_3.json"},
    "Grade 4": {"id": "4", "filename": "math_grade_4.json"},
    "Grade 5": {"id": "5", "filename": "math_grade_5.json"},
    "Grade 6": {"id": "6", "filename": "math_grade_6.json"},
    "Grade 7": {"id": "7", "filename": "math_grade_7.json"},
    "Grade 8": {"id": "8", "filename": "math_grade_8.json"}
}

doc = fitz.open(pdf_path)
grade_content = {grade_info['id']: "" for grade_info in grade_triggers.values()}
current_grade_id = None

print("Reading and sorting content by grade...")
for page in doc:
    text = page.get_text()
    for trigger, grade_info in grade_triggers.items():
        if text.strip().startswith(trigger):
            current_grade_id = grade_info['id']
            break
    if current_grade_id:
        grade_content[current_grade_id] += text + "\n"

print("Saving content to individual JSON files...")
for grade_id, content in grade_content.items():
    if content:
        # Find the correct filename from the trigger dictionary
        filename = next((info['filename'] for info in grade_triggers.values() if info['id'] == grade_id), None)
        if filename:
            file_path = os.path.join(output_dir, filename)
            # Create the structured JSON object
            json_output = {
                "content": content.strip(),
                "grade": grade_id
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2)
            print(f"Saved {file_path}")

doc.close()
print(f"Processing complete. JSON files are in '{output_dir}'.")