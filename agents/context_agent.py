import logging
import json
import os
from typing import Dict, List
from workflow.state import GraphState
from data.vector_store import VectorStore

logger = logging.getLogger(__name__)

sub = {
    'Business Studies': 'bst',
    'Economics': 'eco',
    'Maths-Core': 'math'
}

class ContextAgent:
    def __init__(self, subject, vector_store):
        pyq_path = f"knowledge_base/pyq/pyqs/{sub[subject]}/CUET_{sub[subject]}_pyq_topicwise.json"
        mock_path = f"knowledge_base/pyq/mocks/{sub[subject]}/mock_questions.json"
        self.vector_store = vector_store
        self.pyq_path = os.path.join(os.getcwd(), pyq_path)
        self.mock_path = os.path.join(os.getcwd(), mock_path)
        self.pyq_data = self._load_pyq_data()
        self.mock_data = self._load_mock_data()
        if subject == 'Business Studies':
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
        elif subject == "Economics":
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
        elif subject == "Maths-Core":
            self.dict = {
                          "Relations and Functions": 3,
                          "Algebra": 4,
                          "Calculus": 6,
                          "Vectors and Three-Dimensional Geometry": 3,
                          "Linear Programming": 2,
                          "Probability": 3,
                          "Differential Equations": 3
                        }
        
    def _load_pyq_data(self) -> Dict:
        """Load the structured PYQ data from JSON file"""
        try:
            if os.path.exists(self.pyq_path):
                with open(self.pyq_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"PYQ data file not found: {self.pyq_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading PYQ data: {e}")
            return {}

    def _load_mock_data(self) -> Dict:
        try:
            if os.path.exists(self.mock_path):
                with open(self.mock_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Mock data file not found: {self.mock_path}")
                return {}
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
        """Get context for the current topic from vector store and previous questions"""
        if not state["remaining_topics"]:
            logger.warning("No topics remaining to retrieve context for")
            return state
        
        current_topic = state["remaining_topics"][0]
        logger.info(f"Retrieving context for topic: {current_topic}")
        
        # Get examples from PYQ data
        examples = self._retrieve_pyq_examples(current_topic)
        logger.info(f"Retrieved {len(examples)} examples from PYQ data for {current_topic}")
        
        # If we don't have enough examples, try to supplement with vector store
        if not examples or len(examples) < 3:
            logger.info(f"Found only {len(examples)} examples in PYQ data, supplementing with vector store")
            
            try:
                # Get related questions from vector store
                vs_examples = self._retrieve_from_vector_store(current_topic)
                logger.info(f"Retrieved {len(vs_examples)} examples from vector store for {current_topic}")
                
                # Combine PYQ examples with vector store examples
                examples = self._deduplicate_examples(examples + vs_examples)
                
                # Ensure we don't have too many examples
                if len(examples) > 10:
                    # Keep the most relevant examples
                    examples = examples[:10]
                    
            except Exception as e:
                logger.error(f"Error retrieving examples from vector store: {e}")
        
        # If we still don't have examples, create a minimal placeholder
        if not examples:
            logger.warning(f"No examples found for {current_topic}. Using placeholder examples.")
            examples = [
                f"Example question about {current_topic} (placeholder)",
                f"Another example question about {current_topic} (placeholder)"
            ]
        
        logger.info(f"Final count: {len(examples)} example questions for {current_topic}")
        
        # Get explanatory text if available
        explanations = self._retrieve_explanations(current_topic)
        
        # Update state with context for this topic
        context = state.get("context", {})
        context[current_topic] = {
            "examples": examples,
            "explanations": explanations
        }
        
        return {
            "context": context,
            # Keep remaining_topics unchanged
            "remaining_topics": state["remaining_topics"]
        }
        
    def _deduplicate_examples(self, examples):
        """Remove duplicates and very similar examples"""
        if not examples:
            return []
            
        unique_examples = []
        example_texts = set()
        
        for example in examples:
            # Create a simplified version for comparison
            # Remove whitespace, lowercase, etc.
            simplified = ' '.join(example.split()).lower()
            
            # Check if this is sufficiently different from existing examples
            is_unique = True
            for existing in example_texts:
                # If more than 70% similar, consider it a duplicate
                if self._similarity(simplified, existing) > 0.7:
                    is_unique = False
                    break
                    
            if is_unique:
                unique_examples.append(example)
                example_texts.add(simplified)
                
        return unique_examples
        
    def _similarity(self, text1, text2):
        """Simple text similarity check"""
        # Use Jaccard similarity on word sets
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _retrieve_pyq_examples(self, topic: str) -> List[str]:
        """Retrieve examples for a topic from structured PYQ data"""
        # Reuse existing functionality
        pyq_context = self._get_examples_from_pyq(topic, n_results=(3 * self.dict[topic]))
        return pyq_context["examples"]
        
    def _retrieve_from_vector_store(self, topic: str) -> List[str]:
        """Retrieve examples from vector store"""
        results = self.vector_store.query_collection(query_text=topic, n_results=5)
        return results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else []
        
    def _retrieve_explanations(self, topic: str) -> List[str]:
        """Retrieve explanations from vector store"""
        results = self.vector_store.query_collection(query_text=topic, n_results=5)
        explanations = []
        
        if results["metadatas"] and len(results["metadatas"]) > 0:
            for metadata in results["metadatas"][0]:
                if isinstance(metadata, dict) and "explanation" in metadata:
                    explanations.append(metadata["explanation"])
                    
        return explanations