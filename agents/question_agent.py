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
            with open(f'knowledge_base/{subject}.json', "r", encoding="utf-8") as file:
                data = json.load(file)
            for chapter in data["Chapter"]:
                if chapter["Name"] == name:
                    texts = chapter["text"]
                    selected_texts = random.sample(texts, min(n, len(texts)))  # Ensure N does not exceed available texts
                    NCERT_text = "\n\n".join([item["content"] for item in selected_texts])
                    return NCERT_text
            return None
        NCERT_text = str(n_chunking(current_topic, self.subject, target_count))

        prompt = f"""
        Generate Business Studies exam questions about the topic/subject asked for by the user, following CUET exam patterns.
    
        Guidelines:
        1. Include a mix of question types:
           - Multiple Choice Questions (40%)
           - Match the following questions (20%)
           - Arrange in correct order questions (20%)
           - Statement-based questions (20%) (e.g., "Which statements are correct/incorrect about...")
        
        2. Match the difficulty level of previous examples
        
        3. For each question, provide:
           - Clear question text
           - Answer options 
           - Correct answer
           - Brief explanation justifying the answer
           
        4. Format consistently with the examples provided
        
        5. IMPORTANT: Use NCERT textbook language, terminology and phrasing exactly as it appears in the source text
        
        6. Questions and options MUST use the same vocabulary, definitions, and expressions found in NCERT materials
        
        7. Avoid introducing non-NCERT terms or alternative wording that doesn't match the textbook
        
        8. When referring to concepts, use the exact terminology from the NCERT text
        
        9. For "Match the following" questions:
           - Create two columns with related items
           - Provide options with different combinations of matches
           - Clearly indicate the correct matching
        
        10. For "Arrange in order" questions:
            - Focus on processes, steps, or sequences from the curriculum
            - List items that need to be arranged in correct order
            - Provide options with different possible sequences
        
        11. For statement-based questions:
            - Include 3-4 statements about a concept
            - Ask which statements are correct or incorrect
            - Options should be combinations like "1 and 3 only", "All except 2", etc.
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
            response = openai.chat.completions.create(
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
