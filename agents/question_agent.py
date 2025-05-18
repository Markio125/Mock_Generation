import logging
import openai
from typing import Dict
from workflow.state import GraphState
import config
from utils.token_tracker import TokenTracker
from knowledge_base.chunk_selector import ChunkSelector
import json
import random
import os

logger = logging.getLogger(__name__)

class QuestionAgent:
    def __init__(self, subject, token_tracker=None):
        self.token_tracker = token_tracker or TokenTracker()
        self.subject = subject

    def generate_questions(self, state: GraphState) -> Dict:
        """Generate questions for the current topic"""
        if not state["remaining_topics"]:
            logger.warning("No remaining topics to process")
            return state

        current_topic = state["remaining_topics"][0]
        target_count = state["distribution"][current_topic]
        context = state["context"].get(current_topic, {"examples": [], "explanations": []})

        logger.info(f"Generating {target_count} questions for topic: {current_topic}")

        # Build a prompt with examples if available
        example_text = "\n\n".join(context['examples'][:3]) if context['examples'] else "No examples available"
        example = context["examples"][: (3 * target_count)] if context['examples'] else ["No Examples"] * (3 * target_count)

        def n_chunking(name, subject, n):
            # Construct the file path using lowercase subject name
            # Replace spaces with underscores to match file naming conventions
            file_path = f'knowledge_base/{subject.lower().replace(" ", "_")}.json'
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            for chapter in data["Chapter"]:
                if chapter["Name"] == name:
                    texts = chapter["text"]
                    selected_texts = random.sample(texts, min(n, len(texts)))  # Ensure N does not exceed available texts
                    NCERT_text = "\n\n".join([item["content"] for item in selected_texts])
                    return NCERT_text
            return None
        NCERT_text = str(n_chunking(current_topic, self.subject, target_count))

        # Determine appropriate prompt based on subject
        if self.subject == "Maths-Core" or self.subject == "Maths-Applied":
            prompt = f"""
            Generate {self.subject} exam questions about {current_topic} following CUET exam patterns.
        
            Guidelines:
            1. Include the following question types in the specified ratio:
               - Multiple Choice Questions (60%) - Questions with 4 options where only one is correct
               - Numerical Answer Type (20%) - Problems requiring calculation with a numerical answer
               - Assertion-Reason Questions (20%) - Two statements where students evaluate their truth and relationship
            
            2. For EACH question:
               - Use clear, concise mathematical language appropriate for CUET exam level
               - For multiple choice, provide exactly FOUR answer options (A, B, C, D)
               - Include the correct answer clearly marked
               - Add a brief solution showing the mathematical approach
            
            3. Ensure NCERT textbook accuracy:
               - Use standard mathematical notation and terminology
               - Include a mix of concept-testing and calculation-based problems
               - Questions should cover both theory and application
               
            4. Format each question as follows:
               Question: [Clear mathematical question]
               A. [Option A]
               B. [Option B]
               C. [Option C]
               D. [Option D]
               Answer: [Letter of correct option or numerical answer]
               Solution: [Step-by-step mathematical solution]
               
            5. For numerical answer type questions:
               - Provide a clear problem statement
               - Show the complete step-by-step solution
               - Include the final answer with appropriate units if applicable
            """
        elif self.subject == "General Aptitude":
            prompt = f"""
            Generate {self.subject} exam questions about {current_topic} following CUET exam patterns.
        
            Guidelines:
            1. Include the following question types in the specified ratio:
               - Multiple Choice Questions (50%) - Questions with 4 options where only one is correct
               - Logical Reasoning Questions (30%) - Problems that test reasoning abilities 
               - Data Interpretation Questions (20%) - Questions based on data sets like tables, graphs, etc.
            
            2. For EACH question:
               - Use clear, concise language appropriate for aptitude testing
               - For multiple choice, provide exactly FOUR answer options (A, B, C, D)
               - Include the correct answer clearly marked
               - Add a brief explanation of the solution approach
            
            3. Ensure aptitude testing standards:
               - Questions should be challenging but solvable within 2-3 minutes
               - Include a mix of easy, medium, and difficult questions
               - Ensure questions are unambiguous with only one correct answer
               
            4. Format each question as follows:
               Question: [Clear aptitude question]
               A. [Option A]
               B. [Option B]
               C. [Option C]
               D. [Option D]
               Answer: [Letter of correct option]
               Solution: [Explanation of the correct approach]
            """
        elif self.subject == "English":
            prompt = f"""
            Generate {self.subject} exam questions about {current_topic} following CUET exam patterns.
        
            Guidelines:
            1. Include the following question types in the specified ratio:
               - Reading Comprehension (35%) - Brief passages followed by questions
               - Vocabulary and Grammar MCQs (35%) - Multiple choice questions on language use
               - Writing Skills (30%) - Questions that test writing ability through paragraph completion, error identification, etc.
            
            2. For EACH question:
               - Use clear, standard English appropriate for university entrance tests
               - For multiple choice, provide exactly FOUR answer options (A, B, C, D)
               - Include the correct answer clearly marked
               - Add a brief explanation for why the answer is correct
            
            3. Ensure English language standards:
               - Use contextually rich examples that test genuine language comprehension
               - Include questions on both functional and literary language use
               - For reading comprehension, include diverse text types (expository, narrative, argumentative)
               
            4. Format each question as follows:
               Question: [Clear English language question]
               A. [Option A]
               B. [Option B]
               C. [Option C]
               D. [Option D]
               Answer: [Letter of correct option]
               Explanation: [Brief justification for the correct answer]
            """
        elif self.subject == "Accountancy":
            prompt = f"""
            Generate {self.subject} exam questions about {current_topic} following CUET exam patterns.
        
            Guidelines:
            1. Include the following question types in the specified ratio:
               - Conceptual MCQs (30%) - Questions testing accounting concepts
               - Numerical/Problem Solving (40%) - Questions requiring calculations
               - Application Based Questions (30%) - Questions applying accounting principles to situations
            
            2. For EACH question:
               - Use clear accounting terminology and standard formats
               - For multiple choice, provide exactly FOUR answer options (A, B, C, D)
               - Include the correct answer clearly marked
               - Add a brief explanation or calculation showing the solution
            
            3. Ensure accounting standards:
               - Questions should follow current accounting principles and standards
               - Include a mix of theoretical and practical questions
               - Ensure numerical questions have clear steps and working
               
            4. Format each question as follows:
               Question: [Clear accounting question]
               A. [Option A]
               B. [Option B]
               C. [Option C]
               D. [Option D]
               Answer: [Letter of correct option]
               Solution: [Step-by-step solution or explanation]
            """
        else:
            prompt = f"""
            Generate {self.subject} exam questions about {current_topic} following CUET exam patterns.
        
            Guidelines:
            1. Include the following question types in the specified ratio:
               - Multiple Choice Questions (40%) - Questions with 4 options where only one is correct
               - Match the following questions (20%) - Create two columns with related items to be matched
               - Arrange in correct order questions (20%) - Items that need to be arranged in sequence
               - Statement-based questions (20%) - Include 3-4 statements with options like "1 and 3 only"
            
            2. For EACH question:
               - Use clear, concise language appropriate for CUET exam level
               - Provide exactly FOUR answer options (A, B, C, D) 
               - Include the correct answer clearly marked
               - Add a brief explanation justifying the correct answer
            
            3. Ensure NCERT textbook accuracy:
               - Use EXACTLY the same terminology, definitions, and phrasing as in NCERT materials
               - Avoid introducing terminology not found in standard textbooks
               - Questions should reflect standard NCERT concepts and principles
               
            4. Format each question as follows:
               Question: [Clear question text]
               A. [Option A]
               B. [Option B]
               C. [Option C]
               D. [Option D]
               Answer: [Letter of correct option]
               Explanation: [Brief justification for the correct answer]
               
            5. For each question type, follow these specific guidelines:
               
               FOR MATCH THE FOLLOWING:
               - Create two columns with 4-5 related items
               - Options should be different combinations of matches
               - Clearly indicate correct matching
               
               FOR ARRANGE IN ORDER:
               - Choose processes, steps, or sequences from the curriculum
               - Include 4-5 items to be arranged
               - Provide options with different possible sequences
               
               FOR STATEMENT-BASED QUESTIONS:
               - Include exactly 4 statements about a concept
               - Options should be combinations like "1 and 3 only"
            """

        # Convert example lists to strings with proper formatting
        example_str_1 = "\n\n".join(example[0:(target_count - 1)]) if isinstance(example[0], str) else "No examples available"
        example_str_2 = "\n\n".join(example[(target_count - 1):((2 * target_count) - 1)]) if isinstance(example[0], str) else "No examples available"
        example_str_3 = "\n\n".join(example[((2 * target_count) - 1):(3 * target_count)]) if isinstance(example[0], str) else "No examples available"

        messages = [
            {"role": "system", "content": prompt},

            # First few-shot examples
            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count - 1} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count - 1}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example_str_1},

            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example_str_2},

            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count + 1} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count + 1}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example_str_3},

            # Now adding context by referring to past responses
            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count}, Topic: {current_topic}"""},
        ]

        try:
            response = openai.ChatCompletion.create(
                model=config.GPT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            if self.token_tracker:
                self.token_tracker.update(response)

            generated = response.choices[0].message.content.split('\n\n')
            logger.info(f"Generated {len(generated)} questions for {current_topic}")

            return {
                "questions": {**state.get("questions", {}), current_topic: generated},
                "remaining_topics": state["remaining_topics"][1:],
                "detected_topics": state["detected_topics"]
            }
        except Exception as e:
            logger.error(f"Error generating questions for {current_topic}: {e}")
            # Continue with remaining topics rather than failing completely
            return {
                "questions": {**state.get("questions", {}), current_topic: []},
                "remaining_topics": state["remaining_topics"][1:],
                "detected_topics": state["detected_topics"]
            }
