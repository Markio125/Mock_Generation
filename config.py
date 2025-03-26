import os
import logging
from dotenv import load_dotenv
load_dotenv()
# API Keys and Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

# Database Settings
CHROMA_DB_PATH = "./bs_question_db"
COLLECTION_NAME = "business_studies"

# Default Topics (used as fallback)
DEFAULT_TOPICS = {
    "Nature and Significance of Management": 8,
    "Principles of Management": 8,
    "Business Environment": 7,
    "Planning": 7,
    "Organising": 7,
    "Staffing": 7,
    "Directing": 8,
    "Controlling": 8,
    "Financial Management": 8,
    "Marketing": 12,
    "Consumer Protection": 8
}