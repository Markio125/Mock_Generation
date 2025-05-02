import logging
import openai
import random
from typing import Dict, List
import json
import os
import sys

# Add the project root to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow.state import GraphState
import config
from utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

class CaseQuestionAgent:
    def __init__(self, subject, token_tracker=None, case_studies_per_paper=2, questions_per_case=5):
        self.subject = subject
        self.token_tracker = token_tracker or TokenTracker()
        self.case_studies_per_paper = case_studies_per_paper
        self.questions_per_case = questions_per_case
        # Load example case studies from PYQ
        self.example_case_studies = self._load_pyq_case_studies()

    def generate_case_studies(self, state: GraphState) -> Dict:
        """Generate case study questions based on completed topics"""
        logger.info(f"Generating {self.case_studies_per_paper} case studies with {self.questions_per_case} questions each")
        
        # Select random topics from the detected topics for case studies
        if self.subject == "business studies":
            available_topics = config.DEFAULT_TOPIC_BST.keys()
        else:
            available_topics = config.DEFAULT_TOPIC_ECO.keys()
 
        case_study_topics = random.sample(available_topics, 
                                         min(self.case_studies_per_paper, len(available_topics)))
        
        case_studies = []
        
        for topic in case_study_topics:
            # Get context for the topic
            context = state["context"].get(topic, {"examples": [], "explanations": []})
            
            # Sample NCERT text for the topic
            ncert_text = self._get_topic_text(topic)
            
            # Generate case study
            case_study = self._generate_single_case_study(topic, ncert_text, context)
            if case_study:
                case_studies.append(case_study)
        
        logger.info(f"Successfully generated {len(case_studies)} case studies")
        
        
        return case_studies
    
    def _get_topic_text(self, topic_name: str) -> str:
        """Retrieve text content for a specific topic"""
        try:
            with open('../knowledge_base/business_studies.json', "r", encoding="utf-8") as file:
                data = json.load(file)
                
            for chapter in data["Chapter"]:
                if chapter["Name"] == topic_name:
                    texts = chapter["text"]
                    # Select a few paragraphs to base the case study on
                    selected_texts = random.sample(texts, min(5, len(texts)))
                    return "\n\n".join([item["content"] for item in selected_texts])
            
            return "No topic text found"
        except Exception as e:
            logger.error(f"Error retrieving text for topic {topic_name}: {e}")
            return "Error retrieving topic text"
    
    def _load_pyq_case_studies(self) -> List[Dict]:
        """Load case study examples from PYQ data"""
        try:
            with open('../knowledge_base/pyq/pyqs/bst/CUET_bst_pyq_topicwise.json', 'r', encoding='utf-8') as file:
                pyq_data = json.load(file)
            
            case_studies = []
            
            # Extract case studies from sections 10 and 11
            for section in pyq_data.get('sections', []):
                if section.get('sectionNumber') in [10, 11] and 'caseStudy' in section:
                    case_study = section['caseStudy']
                    questions = section.get('questions', [])
                    
                    # Format questions for the example
                    formatted_questions = []
                    for q in questions:
                        q_text = f"{q.get('questionNumber')}. {q.get('questionText')}"
                        options = q.get('options', [])
                        if isinstance(options, list):
                            option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                        else:  # Handle dict format
                            option_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
                        
                        answer = f"Answer: {chr(64 + q.get('correct_answer')) if isinstance(q.get('correct_answer'), int) else q.get('correct_answer')}"
                        formatted_questions.append(f"{q_text}\n{option_text}\n{answer}")
                    
                    case_studies.append({
                        'title': case_study.get('title', ''),
                        'text': case_study.get('text', ''),
                        'questions': "\n\n".join(formatted_questions),
                        'topic': questions[0].get('topic') if questions else ''
                    })
            
            return case_studies
        except Exception as e:
            logger.error(f"Error loading PYQ case studies: {e}")
            return []
    
    def _generate_single_case_study(self, topic: str, ncert_text: str, context: Dict) -> Dict:
        """Generate a single case study with questions"""
        prompt = f"""
        Create a case study with {self.questions_per_case} multiple-choice questions about {topic}.
        
        Guidelines:
        1. Begin with a realistic business scenario (200-300 words) related to {topic}
        2. The scenario should be based on the NCERT content provided
        3. After the scenario, create {self.questions_per_case} multiple-choice questions that:
           - Test comprehension and application of business concepts
           - Relate directly to the case study content
           - Follow CUET exam patterns
           
        4. For each question, provide:
           - Clear question text referencing the case
           - Four answer options (A, B, C, D)
           - Correct answer
           - Brief explanation justifying the answer
           
        5. IMPORTANT: Use NCERT textbook terminology and concepts exactly as they appear
           
        6. Format your response as follows:
        
        CASE STUDY: [Title]
        
        [Case study text here]
        
        QUESTIONS:
        
        1. [Question text]
        A. [Option A]
        B. [Option B]
        C. [Option C]
        D. [Option D]
        Answer: [Correct option letter]
        Explanation: [Brief explanation]
        
        2. [Question text]
        ...and so on
        """
        
        # Create messages including case study examples from PYQs
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        # Add example case studies from PYQs as examples
        for i, example in enumerate(self.example_case_studies):
            messages.append({"role": "system", "content": f"Here's Example {i+1} of a CUET case study format:"})
            messages.append({"role": "system", "content": f"""
            CASE STUDY: {example['title']}
            
            {example['text']}
            
            QUESTIONS:
            
            {example['questions']}
            """})
        
        # Add the current request
        messages.append({"role": "user", "content": f"""Use the below provided NCERT text as the information base for creating 
        a case study on {topic} with {self.questions_per_case} questions.
        
        NCERT TEXT:
        -------------------------------------------------------------------------
        {ncert_text}
        -------------------------------------------------------------------------
        """})
        
        try:
            response = openai.chat.completions.create(
                model=config.GPT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2500
            )
            
            if self.token_tracker:
                self.token_tracker.update(response)
                
            content = response.choices[0].message.content
            
            # Parse the case study and questions
            parts = content.split("QUESTIONS:")
            if len(parts) < 2:
                logger.warning(f"Unexpected response format for case study on {topic}")
                return None
                
            case_text = parts[0].strip()
            questions_text = parts[1].strip()
            
            # Clean up case text to extract title and content
            case_parts = case_text.split("\n\n", 1)
            title = case_parts[0].replace("CASE STUDY:", "").strip() if len(case_parts) > 1 else "Case Study"
            content = case_parts[1].strip() if len(case_parts) > 1 else case_parts[0].strip()
            
            return {
                "topic": topic,
                "title": title,
                "content": content,
                "questions": questions_text
            }
            
        except Exception as e:
            logger.error(f"Error generating case study for {topic}: {e}")
            return None

if __name__ == "__main__":
    # Quick test for CaseQuestionAgent
    from agents.case_q_agent import CaseQuestionAgent
    # Create a dummy “GraphState‐like” object with only the expected key
    state = {"context": {}}
    agent = CaseQuestionAgent()
    case_studies = agent.generate_case_studies(state)
    import json
    print(json.dumps(case_studies, indent=2))
