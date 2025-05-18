import logging
import json
from typing import List, Dict
import openai
import config
from utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

class TopicExtractor:
    def __init__(self, token_tracker=None):
        self.token_tracker = token_tracker or TokenTracker()
        
    def extract_topics(self, corpus: List[Dict]) -> List[str]:
        """Automatically detect topics from question corpus"""
        questions = [q["question"] for q in corpus if "question" in q]
        if not questions:
            logger.warning("No questions found in corpus")
            return config.DEFAULT_TOPICS
        
        prompt = f"""
        Analyze these Business Studies questions and extract main topics:
        {questions[:20]}  # Limiting to prevent token limits
        
        Return as JSON array of topic names (maximum 10 topics)
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=config.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            if self.token_tracker:
                self.token_tracker.update(response)
                
            response_text = response.choices[0].message.content
            topics = json.loads(response_text)["topics"]
            logger.info(f"Extracted {len(topics)} topics: {topics}")
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            logger.info("Using default topics instead")
            return config.DEFAULT_TOPICS
