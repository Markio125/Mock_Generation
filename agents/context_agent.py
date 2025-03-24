import logging
from typing import Dict
from workflow.state import GraphState
from data.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ContextAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def retrieve_context(self, state: GraphState) -> Dict:
        """Retrieve context for the current topic"""
        if not state["remaining_topics"]:
            logger.warning("No remaining topics to process")
            return state
            
        current_topic = state["remaining_topics"][0]
        logger.info(f"Retrieving context for topic: {current_topic}")
        
        try:
            # Retrieve similar questions and explanations
            results = self.vector_store.query_collection(query_text=current_topic, n_results=5)
            
            examples = results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else []
            explanations = []
            
            if results["metadatas"] and len(results["metadatas"]) > 0:
                for metadata in results["metadatas"][0]:
                    if isinstance(metadata, dict) and "explanation" in metadata:
                        explanations.append(metadata["explanation"])
            
            context = {
                "examples": examples,
                "explanations": explanations
            }
            
            logger.info(f"Retrieved {len(examples)} example questions for {current_topic}")
            return {
                "context": {**state.get("context", {}), current_topic: context},
                "detected_topics": state["detected_topics"]
            }
        except Exception as e:
            logger.error(f"Error retrieving context for {current_topic}: {e}")
            # Return empty context to continue the workflow
            return {
                "context": {**state.get("context", {}), current_topic: {"examples": [], "explanations": []}},
                "detected_topics": state["detected_topics"]
            }

