import logging
import numpy as np
from typing import Dict
from workflow.state import GraphState
from data.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DistributionAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def analyze_distribution(self, state: GraphState) -> Dict:
        """Probabilistic distribution based on topic prevalence"""
        logger.info("Analyzing question distribution across topics")
        
        try:
            # Calculate topic weights using question frequency
            topic_weights = {}
            for topic in state["detected_topics"]:
                try:
                    results = self.vector_store.query_collection(query_text=topic, n_results=10)
                    topic_weights[topic] = len(results["ids"][0]) if results["ids"] and results["ids"][0] else 1
                except Exception as e:
                    logger.error(f"Error querying for topic '{topic}': {e}")
                    topic_weights[topic] = 1  # Default weight
            
            # If no weights were found, use uniform distribution
            if not topic_weights or sum(topic_weights.values()) == 0:
                logger.warning("No topic weights found, using uniform distribution")
                equal_weight = 1.0 / len(state["detected_topics"])
                probabilities = {t: equal_weight for t in state["detected_topics"]}
            else:
                # Normalize weights
                total = sum(topic_weights.values())
                probabilities = {k: v/total for k, v in topic_weights.items()}
            
            # Ensure we have at least one question per topic
            total_questions = state["total_questions"]
            min_questions = len(probabilities)
            
            if total_questions < min_questions:
                logger.warning(f"Too few total questions ({total_questions}) for {min_questions} topics. Setting 1 per topic.")
                distribution = {t: 1 for t in probabilities.keys()}
            else:
                # Multinomial distribution for remaining questions after ensuring minimum
                remaining = total_questions - min_questions
                if remaining > 0:
                    counts = np.random.multinomial(remaining, list(probabilities.values()))
                    distribution = {t: c + 1 for t, c in zip(probabilities.keys(), counts)}
                else:
                    distribution = {t: 1 for t in probabilities.keys()}
            
            logger.info(f"Distribution calculated: {distribution}")
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