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
    "Nature and Significance of Management": 4,  # 8%
    "Principles of Management": 4,  # 8%
    "Business Environment": 4,  # 8%
    "Planning": 6,  # 12%
    "Organising": 5,  # 10%
    "Staffing": 5,  # 10%
    "Directing": 3,  # 6%
    "Controlling": 3,  # 6%
    "Financial Management": 9,  # 18%
    "Marketing": 5,  # 10%
    "Consumer Protection": 3,  # 6% 
}