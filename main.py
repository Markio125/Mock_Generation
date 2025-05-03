# =====================
import logging
import json
import openai
import config
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
            detected_topics = config.DEFAULT_TOPIC_BST.keys()  #topic_extractor.extract_topics(corpus)
        else:
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
            for topic in detected_topics:
                try:
                    prompt = f"Generate 5 Business Studies exam MCQ questions for the CUET exam about {topic}."
                    response = openai.chat.completions.create(
                        model=config.GPT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    
                    token_tracker.update(response)
                    
                    generated = response.choices[0].message.content.split('\n\n')
                    final_paper[topic] = generated

                except Exception as gen_error:
                    logger.error(f"Error in fallback generation for {topic}: {gen_error}")
                    final_paper[topic] = [f"Example question about {topic}"]

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
    output_path = "outputs/generated_paper.json"
    subject = ['Business Studies', 'Economics']
    main(corpus_path, output_path, subject[0]) #put a value in the [] of subject depending on which sub is needed