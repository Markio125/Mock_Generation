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
DEFAULT_TOPICS = ["Business Management", "Marketing", "Finance", "Operations"]