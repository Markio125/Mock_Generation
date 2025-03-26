import os
import json
import re
from PyPDF2 import PdfReader

def read_pdfs_from_folder(folder_path):
    pdf_contents = {}
    
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found")
        return pdf_contents
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        try:
            reader = PdfReader(file_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            pdf_contents[pdf_file] = text
            
        except Exception as e:
            print(f"Error reading {pdf_file}: {str(e)}")
    
    return pdf_contents

def extract_questions(text):
    # Split into potential question blocks
    question_blocks = re.split(r'\n\s*\d+\.', text)
    questions = []
    
    for block in question_blocks[1:]:  # Skip first split which is usually empty
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if len(lines) < 5:  # Need at least question + 4 options
            continue
            
        question_dict = {
            "question": lines[0].strip(),
            "options": [],
            "correct_answer": ""
        }
        
        # Extract options
        for line in lines[1:]:
            if re.match(r'^\([a-d]\)', line.lower()):
                question_dict["options"].append(line.strip())
                
        # Only add if we found exactly 4 options
        if len(question_dict["options"]) == 4:
            questions.append(question_dict)
            
    return questions

def convert_to_json(pdf_contents, output_path):
    all_questions = []
    
    for filename, content in pdf_contents.items():
        questions = extract_questions(content)
        all_questions.extend(questions)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=4)
    
    print(f"Total questions extracted: {len(all_questions)}")
    return all_questions

if __name__ == "__main__":
    folder_path = 'pyqs/bst'
    output_path = 'pyqs/questions.json'
    pdf_contents = read_pdfs_from_folder(folder_path)
    print(f"Successfully read {len(pdf_contents)} PDF files")
    
    if pdf_contents:
        questions = convert_to_json(pdf_contents, output_path)