# =====================
import logging
import json
import openai
import config
import time
import sys
from utils.logging_utils import setup_logger
from utils.token_tracker import TokenTracker
from data.vector_store import VectorStore
from data.topic_extractor import TopicExtractor
from agents.distribution_agent import DistributionAgent
from agents.context_agent import ContextAgent
from agents.question_agent import QuestionAgent
from workflow.graph_builder import WorkflowBuilder
from agents.case_q_agent import CaseQuestionAgent

# Setup logger
logger = setup_logger()

def print_progress(message, current, total=None):
    """Display a progress message with optional progress bar"""
    if total:
        progress = int(30 * current / total)
        bar = "█" * progress + "-" * (30 - progress)
        percentage = int(100 * current / total)
        sys.stdout.write(f"\r{message} [{bar}] {percentage}% ({current}/{total})")
    else:
        sys.stdout.write(f"\r{message}")
    sys.stdout.flush()

def main(corpus_path: str, output_path: str, subject, total_questions: int = 50):
    """Run the complete workflow"""
    # Initialize components
    openai.api_key = config.OPENAI_API_KEY
    token_tracker = TokenTracker()
    vector_store = VectorStore()
    # topic_extractor = TopicExtractor(token_tracker)
    
    # Initialize final_paper to a default value
    final_paper = {}
    
    try:
        # Load question papers
        logger.info(f"Loading corpus from {corpus_path}")
        try:
            with open(corpus_path, encoding="utf-8") as f:
                corpus = json.load(f)
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            logger.info("Using empty corpus instead")
            corpus = []
        
        # Preprocess data
        if subject == "Business Studies":
            detected_topics = config.DEFAULT_TOPIC_BST.keys()
        elif subject == "Maths-Core":
            detected_topics = config.DEFAULT_TOPIC_MATH.keys()
        elif subject == "Maths-Applied":
            detected_topics = config.DEFAULT_TOPIC_MAPP.keys()
        elif subject == "General Aptitude":
            detected_topics = config.DEFAULT_TOPIC_GENAP.keys()
        elif subject == "English":
            detected_topics = config.DEFAULT_TOPIC_ENG.keys()
        elif subject == "Accountancy":
            detected_topics = config.DEFAULT_TOPIC_ACCT.keys()
        else:  # Economics
            detected_topics = config.DEFAULT_TOPIC_ECO.keys()
        
        # Handle potential ChromaDB dimension issues
        try:
            vector_store.initialize_from_corpus(corpus)
        except Exception as e:
            logger.error(f"Error in vector store initialization: {e}")
            logger.info("Attempting to recreate collection...")
            try:
                vector_store.get_or_create_collection(force_recreate=True)  # Changed to True
                vector_store.initialize_from_corpus(corpus)
            except Exception as inner_e:
                logger.error(f"Failed to recover from vector store error: {inner_e}")
                # Consider a more graceful degradation here
        
        # Initialize agents
        distribution_agent = DistributionAgent(subject, vector_store)  # Pass vector_store if needed
        context_agent = ContextAgent(subject, vector_store)
        question_agent = QuestionAgent(subject, token_tracker)
        
        # Create workflow
        workflow_builder = WorkflowBuilder(distribution_agent, context_agent, question_agent)
        app = workflow_builder.create_workflow()
        
        inputs = {
            "total_questions": total_questions,
            "detected_topics": detected_topics,
            "context": {},
            "questions": {},
            "remaining_topics": []
        }
        
        logger.info("Starting workflow execution")
        
        try:
            # Execute workflow
            result = app.invoke(inputs)
            
            # Extract questions from the final state
            if isinstance(result, dict) and "questions" in result:
                final_paper = result["questions"]
                logger.info(f"Workflow completed successfully with {len(final_paper)} topics")
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
                final_paper = {topic: [] for topic in detected_topics}
                
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")
            
            logger.info("Using fallback question generation")
            # Fallback to direct question generation without the workflow
            for i, topic in enumerate(detected_topics):
                print_progress(f"Generating fallback questions", i+1, len(detected_topics))
                logger.info(f"Generating fallback questions for topic {i+1}/{len(detected_topics)}: {topic}")
                try:
                    # Determine how many questions to generate based on distribution
                    num_questions = 2  # Default to 2 questions per topic in fallback mode
                    if subject == "Business Studies":
                        if topic in config.DEFAULT_TOPIC_BST:
                            num_questions = min(config.DEFAULT_TOPIC_BST[topic], 3)  # Cap at 3 for fallback
                    elif subject == "Maths-Core":
                        if topic in config.DEFAULT_TOPIC_MATH:
                            num_questions = min(config.DEFAULT_TOPIC_MATH[topic], 3)  # Cap at 3 for fallback
                    elif subject == "Maths-Applied":
                        if topic in config.DEFAULT_TOPIC_MAPP:
                            num_questions = min(config.DEFAULT_TOPIC_MAPP[topic], 3)  # Cap at 3 for fallback
                    elif subject == "General Aptitude":
                        if topic in config.DEFAULT_TOPIC_GENAP:
                            num_questions = min(config.DEFAULT_TOPIC_GENAP[topic], 3)  # Cap at 3 for fallback
                    elif subject == "English":
                        if topic in config.DEFAULT_TOPIC_ENG:
                            num_questions = min(config.DEFAULT_TOPIC_ENG[topic], 3)  # Cap at 3 for fallback
                    elif subject == "Accountancy":
                        if topic in config.DEFAULT_TOPIC_ACCT:
                            num_questions = min(config.DEFAULT_TOPIC_ACCT[topic], 3)  # Cap at 3 for fallback
                    else:
                        if topic in config.DEFAULT_TOPIC_ECO:
                            num_questions = min(config.DEFAULT_TOPIC_ECO[topic], 3)  # Cap at 3 for fallback
                    
                    prompt = f"Generate {num_questions} {subject} exam MCQ questions for the CUET exam about {topic}. Format each question clearly with 4 options (A, B, C, D) and include the correct answer."
                    response = openai.ChatCompletion.create(
                        model=config.GPT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    
                    token_tracker.update(response)
                    
                    generated = response.choices[0].message.content.split('\n\n')
                    final_paper[topic] = generated
                    logger.info(f"Generated {len(generated)} fallback questions for {topic}")

                except Exception as gen_error:
                    logger.error(f"Error in fallback generation for {topic}: {gen_error}")
                    final_paper[topic] = [f"Example question about {topic}"]
            
            # Clear the progress bar after completion
            print_progress("", 0, 1)
            sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line
            sys.stdout.flush()

        case_question_agent = CaseQuestionAgent(subject, token_tracker)
        try:
            logger.info("Generating case studies...")
            case_studies = case_question_agent.generate_case_studies({"context": result.get("context", {})})

            # Add case studies to the final paper
            final_paper["case_studies"] = case_studies
            logger.info(f"Added {len(case_studies)} case studies to the final paper")
        except Exception as case_error:
            logger.error(f"Error generating case studies: {case_error}")
            final_paper["case_studies"] = []

        # Save output
        logger.info(f"Saving generated paper to {output_path}")
        logger.info(f"OpenAI Token Usage: {token_tracker.get_stats()}")
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_paper, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            
        return final_paper
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        # Return empty result rather than raising
        return final_paper

if __name__ == "__main__":
    corpus_path = "processed_papers/1.json"
    subject = ['Business Studies', 'Economics', 'Maths-Core', 'Maths-Applied', 'General Aptitude', 'English', 'Accountancy']
    print("Select a subject:")
    for index, subj in enumerate(subject, start=1):
        print(f"{index}. {subj}")
    
    try:
        subject_choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= subject_choice <= len(subject):
            selected_subject = subject[subject_choice - 1]
        else:
            raise ValueError("Invalid choice. Please enter a number from the list.")
    except ValueError as ve:
        logger.error(f"Invalid input: {ve}")
        selected_subject = subject[0]  # Default to the first subject if input is invalid
    output_path = f"outputs/generated_paper_{selected_subject.lower().replace(' ', '_')}.json"

    main(corpus_path, output_path, selected_subject)