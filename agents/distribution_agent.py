import logging
import numpy as np
from typing import Dict
from workflow.state import GraphState
from data.vector_store import VectorStore
import json

logger = logging.getLogger(__name__)

class DistributionAgent:
    def __init__(self, vector_store=None, default_distribution_path="utils/business_studies_distribution.json"):
        """Initialize the distribution agent.
        
        Args:
            vector_store: Optional vector store for advanced distribution analysis
            default_distribution_path: Default path for the distribution file
        """
        self.vector_store = vector_store
        self.default_distribution_path = default_distribution_path

    @staticmethod
    def question_distribution_manual(file_path: str) -> Dict:
        logger.info("Setting question distribution across topics")
        with open(file_path, "r", encoding="utf-8") as json_file:
            chap_ques = json.load(json_file)
        return {
            "Distribution": chap_ques
        }

    def analyze_distribution(self, state: GraphState, file_path: str = None) -> Dict:
        """Get question distribution from manual config and set up state"""
        file_path = file_path or self.default_distribution_path
        logger.info("Setting question distribution across topics")
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                distribution = json.load(json_file)

            # Ensure compatibility with existing workflow expectations
            return {
                "distribution": distribution,
                "remaining_topics": list(distribution.keys()),
                "detected_topics": state["detected_topics"]
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_distribution: {e}")
            # Fallback to uniform distribution
            equal_dist = state["total_questions"] // len(state["detected_topics"])
            distribution = {t: max(1, equal_dist) for t in state["detected_topics"]}
            return {
                "distribution": distribution,
                "remaining_topics": list(distribution.keys()),
                "detected_topics": state["detected_topics"]
            }

if __name__ == "__main__":
    test = DistributionAgent()
    text = test.question_distribution_manual("utils/business_studies_distribution.json")
    print(text)

