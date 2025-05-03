import json
import random


class ChunkSelector:

    @staticmethod
    def n_chunking(name, subject, n):
        if subject == 'Business Studies':
            with open('business_studies.json', "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            with open('economics.json', "r", encoding="utf-8") as file:
                data = json.load(file)
        for chapter in data["Chapter"]:
            if chapter["Name"] == name:
                texts = chapter["text"]
                selected_texts = random.sample(texts, min(n, len(texts)))  # Ensure N does not exceed available texts
                NCERT_text = "\n\n".join([item["content"] for item in selected_texts])
                return NCERT_text
        return None