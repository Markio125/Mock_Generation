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
    def __init__(self, token_tracker=None):
        self.token_tracker = token_tracker or TokenTracker()

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

        # prompt = f"""
        # Generate {target_count} Business Studies exam questions about {current_topic}.
        #
        # Examples from previous questions:
        # {example_text}
        #
        # Guidelines:
        # 1. Include questions only of Multiple Choice Questions type
        # 2. Match the difficulty level of previous examples
        # 3. For each question, provide:
        #    - Cleaor question text
        #    - Answer options (for multiple choice)
        #    - Correct answer
        #    - Brief explanation justifying the answer
        # 4. Format consistently with the examples provided
        # """knowledge_base\business_studies.json
        def n_chunking(name, n):
            with open('knowledge_base/business_studies.json', "r", encoding="utf-8") as file:
                data = json.load(file)
            for chapter in data["Chapter"]:
                if chapter["Name"] == name:
                    texts = chapter["text"]
                    selected_texts = random.sample(texts, min(n, len(texts)))  # Ensure N does not exceed available texts
                    NCERT_text = "\n\n".join([item["content"] for item in selected_texts])
                    return NCERT_text
            return None
        NCERT_text = str(n_chunking(current_topic, target_count))

        prompt = f"""
        Generate Business Studies exam questions about the topic/subject asked for by the user.

        Guidelines:
        1. Include questions only of Multiple Choice Questions type
        2. Match the difficulty level of previous examples
        3. For each question, provide:
           - Clear question text
           - Answer options (for multiple choice)
           - Correct answer
           - Brief explanation justifying the answer
        4. Format consistently with the examples provided
        """

        messages = [
            {"role": "system", "content": prompt},

            # First few-shot examples
            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count - 1} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count - 1}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example[0: (target_count - 1)]},

            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example[(target_count - 1) : ((2 * target_count) - 1)]},

            {"role": "user", "content": f"""Use the below provided text as information base to prepare the {target_count + 1} questions
        -------------------------------------------------------------------------
        {NCERT_text}
        -------------------------------------------------------------------------
        Number of Questions: {target_count + 1}, Topic: {current_topic}"""},
            {"role": "assistant", "content": example[((2 * target_count) - 1) : (3 * target_count)]},

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
                # messages=[{"role": "user", "content": prompt}],
                messages = messages,
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
