import logging
import json
import os
from typing import Dict, List
from workflow.state import GraphState
from data.vector_store import VectorStore

logger = logging.getLogger(__name__)

sub = {
    'business_studies': 'bst',
    'economics': 'eco'
}

class ContextAgent:
    def __init__(self, subject, vector_store):
        pyq_path = f"knowledge_base/pyq/pyqs/bst/CUET_{sub[subject]}_pyq_topicwise.json"
        mock_path = f"knowledge_base/pyq/mocks/{sub[subject]}/mock_questions.json"
        self.vector_store = vector_store
        self.pyq_path = os.path.join(os.getcwd(), pyq_path)
        self.mock_path = os.path.join(os.getcwd(), mock_path)
        self.pyq_data = self._load_pyq_data()
        self.mock_data = self._load_mock_data()
        if subject == 'business_studies':
            self.dict = {
                            "Nature and Significance of Management": 4,
                            "Principles of Management": 4,
                            "Business Environment": 4,
                            "Planning": 6,
                            "Organising": 5,
                            "Staffing": 5,
                            "Directing": 3,
                            "Controlling": 3,
                            "Financial Management": 9,
                            "Marketing": 5,
                            "Consumer Protection": 3
                        }
        elif subject == "economics":
            self.dict = {
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
        
    def _load_pyq_data(self) -> Dict:
        """Load the structured PYQ data from JSON file"""
        try:
            with open(self.pyq_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading PYQ data: {e}")
            return {}

    def _load_mock_data(self) -> Dict:
        try:
            with open(self.mock_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading Mock data: {e}")
            return {}
            
    def _get_examples_from_pyq(self, topic: str, n_results: int) -> Dict:
        """Retrieve examples for a topic from the structured PYQ data"""

        if not self.pyq_data or 'sections' not in self.pyq_data:
            logger.warning("PYQ data not available or invalid format")
            return {"examples": [], "explanations": []}
            
        examples = []
        explanations = []
        n_results = self.dict[topic]
        
        # Normalize topic for case-insensitive comparison
        normalized_topic = topic.lower().strip()
        
        # Collect all matching questions across sections
        for section in self.pyq_data.get('sections', []):
            for question in section.get('questions', []):
                question_topic = question.get('topic', '').lower().strip()
                
                if normalized_topic in question_topic or question_topic in normalized_topic:
                    # Format the question with options
                    formatted_question = question.get('questionText', '')
                    
                    # Handle regular options format
                    if 'options' in question:
                        for i, option in enumerate(question.get('options', [])):
                            formatted_question += f"\n({chr(65+i)}) {option}"
                    
                    # Handle list format questions (matching type)
                    if 'listI' in question and 'listII' in question:
                        formatted_question += "\nList I:"
                        for key, value in question.get('listI', {}).items():
                            formatted_question += f"\n{key} {value}"
                        
                        formatted_question += "\nList II:"
                        for key, value in question.get('listII', {}).items():
                            formatted_question += f"\n{key} {value}"
                    
                    # Add instruction if present
                    if 'instruction' in question:
                        formatted_question += f"\n{question.get('instruction', '')}"
                    
                    # Get correct answer
                    correct_answer = question.get('correct_answer', '')
                    
                    examples.append(formatted_question)
                    
                    # Create a simple explanation based on the correct answer=
                    # Stop if we've reached the desired number of examples
                    if len(examples) >= n_results:
                        break
            
            if len(examples) >= n_results:
                break

        if len(examples) < n_results:
            for question in self.mock_data.get('questions', []):
                question_topic = question.get('topic', '').lower().strip()
                if normalized_topic in question_topic or question_topic in normalized_topic:
                    formatted_question = question.get('questionText', '')
                    # Handle regular options format
                    if 'options' in question:
                        for i, option in enumerate(question.get('options', [])):
                            formatted_question += f"\n({chr(65 + i)}) {option}"

                    # Handle list format questions (matching type)
                    if 'listI' in question and 'listII' in question:
                        formatted_question += "\nList I:"
                        for key, value in question.get('listI', {}).items():
                            formatted_question += f"\n{key} {value}"

                        formatted_question += "\nList II:"
                        for key, value in question.get('listII', {}).items():
                            formatted_question += f"\n{key} {value}"

                    correct_answer = question.get('correct_answer', '')
                    examples.append(formatted_question)

                    if len(examples) >= n_results:
                        break




        return {
            "examples": examples,
            "explanations": explanations
        }
        
    def retrieve_context(self, state: GraphState) -> Dict:
        """Retrieve context for the current topic, prioritizing PYQ data"""
        if not state["remaining_topics"]:
            logger.warning("No remaining topics to process")
            return state
            
        current_topic = state["remaining_topics"][0]
        logger.info(f"Retrieving context for topic: {current_topic}")
        
        try:
            # First try to get examples from the structured PYQ data
            pyq_context = self._get_examples_from_pyq(current_topic, n_results = (3 * self.dict[current_topic]))
            
            # If we didn't get enough examples from PYQ data, supplement with vector store
            if len(pyq_context["examples"]) < (3 * self.dict[current_topic]):
                logger.info(f"Found only {len(pyq_context['examples'])} examples in PYQ data, supplementing with vector store")
                
                # Retrieve similar questions and explanations from vector store
                results = self.vector_store.query_collection(query_text=current_topic, n_results=5-len(pyq_context["examples"]))
                
                additional_examples = results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else []
                additional_explanations = []
                
                if results["metadatas"] and len(results["metadatas"]) > 0:
                    for metadata in results["metadatas"][0]:
                        if isinstance(metadata, dict) and "explanation" in metadata:
                            additional_explanations.append(metadata["explanation"])
                
                # Combine PYQ examples with vector store examples
                context = {
                    "examples": pyq_context["examples"] + additional_examples,
                    "explanations": pyq_context["explanations"] + additional_explanations
                }
            else:
                context = pyq_context
            
            logger.info(f"Retrieved {len(context['examples'])} example questions for {current_topic}")
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