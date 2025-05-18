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
DEFAULT_TOPIC_BST = {
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

DEFAULT_TOPIC_ECO = {
  "Determination of Income and Employment": 8,
  "Money and Banking": 5,
  "Government Budget and the Economy": 3,
  "Open Economy Macroeconomics": 4,
  "Indian Economic Development": 20,
  "National Income Accounting": 3,
  "Introduction": 3,
  "Theory of Consumer Behaviour": 3,
  "Market Equilibrium": 1
}

DEFAULT_TOPIC_MATH = {
  "Relations and Functions": 3,
  "Algebra": 4,
  "Calculus": 6,
  "Vectors and Three-Dimensional Geometry": 3,
  "Linear Programming": 2,
  "Probability": 3,
  "Differential Equations": 3
}

DEFAULT_TOPIC_MAPP = {
  "Matrices and Determinants": 5,
  "Complex Numbers": 3,
  "Analytical Geometry": 5,
  "Differential Calculus": 7,
  "Integral Calculus": 6,
  "Differential Equations": 4,
  "Statistics and Probability": 5,
  "Mathematical Modeling": 3
}

DEFAULT_TOPIC_GENAP = {
  "Verbal Ability": 6,
  "Logical Reasoning": 8,
  "Quantitative Aptitude": 7,
  "Data Interpretation": 7,
  "General Knowledge": 6,
  "Analytical Reasoning": 6,
  "Decision Making": 5,
  "Computer Literacy": 5
}

DEFAULT_TOPIC_ENG = {
  "Reading Comprehension": 8,
  "Grammar and Usage": 7,
  "Vocabulary and Word Usage": 6,
  "Writing Skills": 8,
  "Literature and Literary Devices": 6,
  "Critical Analysis": 5,
  "Communication Skills": 5,
  "Academic Writing": 5
}

DEFAULT_TOPIC_ACCT = {
  "Financial Statements Analysis": 7,
  "Corporate Accounting": 8,
  "Partnership Accounts": 6,
  "Accounting for Not-for-Profit Organizations": 5,
  "Cash Flow Statements": 6,
  "Computerized Accounting": 4,
  "Accounting Standards": 4,
  "Cost Accounting": 5,
  "Taxation Principles": 5
}