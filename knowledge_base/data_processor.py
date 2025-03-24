import pymupdf
import json
import os
import re
import nltk
from nltk.corpus import stopwords


class PDFExtractor:
    """Extracts and processes text from a PDF."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        """Extracts text from a PDF and returns a structured chapter-wise format."""
        doc = pymupdf.open(self.pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text("text") + "\n"

        cleaned_text = TextProcessor.clean_text(full_text)
        return cleaned_text


class TextProcessor:
    """Handles text cleaning and stopword removal."""

    @staticmethod
    def clean_text(text):
        """Cleans text by removing stopwords, special characters, and extra spaces."""
        if not text:
            return "No text found"

        text = text.lower()

        # Removing only unwanted characters (newlines, extra spaces, and dots)
        text = text.replace("\n", " ")  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[.\-_,;!?]', '', text)  # Remove specific symbols but keep words intact

        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]

        return " ".join(filtered_words)

    @staticmethod
    def split_into_sections(text):
        """Splits text into equal sections based on num_sections."""
        words = text.split()
        num_sections = len(words) // 500   #each section will contain roughly 1000 words
        chunk_size = len(words) // num_sections
        sections = []

        for i in range(num_sections):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_sections - 1 else len(words)
            section_text = " ".join(words[start:end])
            sections.append({"index": i, "content": section_text})

        return sections


class JSONSaver:
    """Handles saving extracted data to a JSON file while keeping old data."""

    def __init__(self, output_path):
        self.output_path = output_path

    def save(self, chapter_name, sections):
        """Updates the existing JSON file with new chapter data or creates a new one."""
        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = {"Chapter": []}
        else:
            existing_data = {"Chapter": []}

        # Append new chapter
        new_chapter = {
            "Name": chapter_name,
            "text": sections
        }
        existing_data["Chapter"].append(new_chapter)

        # Save updated data back to JSON
        with open(self.output_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4, ensure_ascii=False)

        print(f"Updated text saved to {self.output_path}")


class PDFProcessor:
    """Main class to manage the full workflow of extracting and updating JSON."""

    def __init__(self, pdf_path, json_path, chapter_name):
        self.pdf_extractor = PDFExtractor(pdf_path)
        self.json_saver = JSONSaver(json_path)
        self.chapter_name = chapter_name

    def process(self):
        """Extracts text, processes it, and updates JSON with structured chapter data."""
        extracted_text = self.pdf_extractor.extract_text()
        sections = TextProcessor.split_into_sections(extracted_text)
        self.json_saver.save(self.chapter_name, sections)


# Example Usage
pdf_file = "pdf_storage\Chapter 11_ Consumer Protection.pdf"  # Replace with file path
json_file = "business_studies.json"
chapter_name = "Consumer Protection"  # chapter name

processor = PDFProcessor(pdf_file, json_file, chapter_name)
processor.process()