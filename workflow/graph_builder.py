import logging
from langgraph.graph import StateGraph, END
from workflow.state import GraphState
from agents.distribution_agent import DistributionAgent
from agents.context_agent import ContextAgent
from agents.question_agent import QuestionAgent

logger = logging.getLogger(__name__)

class WorkflowBuilder:
    def __init__(self, distribution_agent, context_agent, question_agent):
        self.distribution_agent = distribution_agent
        self.context_agent = context_agent
        self.question_agent = question_agent
        
    def create_workflow(self):
        """Create and configure the workflow graph"""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("analyze_distribution", self.distribution_agent.analyze_distribution)
        workflow.add_node("retrieve_context", self.context_agent.retrieve_context)
        workflow.add_node("generate_questions", self.question_agent.generate_questions)
        
        workflow.set_entry_point("analyze_distribution")
        workflow.add_edge("analyze_distribution", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_questions")
        
        workflow.add_conditional_edges(
            "generate_questions",
            lambda state: "retrieve_context" if state["remaining_topics"] else END,
            {
                "retrieve_context": "retrieve_context",
                END: END
            }
        )
        
        return workflow.compile()
