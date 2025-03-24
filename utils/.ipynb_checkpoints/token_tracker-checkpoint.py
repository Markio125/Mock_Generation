# utils/.py
# =====================
import logging
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

class TokenTracker:
    def __init__(self):
        self.usage = {"input": 0, "output": 0}
    
    def update(self, response):
        """Update token usage from OpenAI response"""
        if hasattr(response, "usage"):
            self.usage["input"] += response.usage.prompt_tokens
            self.usage["output"] += response.usage.completion_tokens
        else:
            logger.warning("Token usage data not found in OpenAI response")
    
    def get_cost_estimate(self):
        """Calculate estimated cost based on current token usage"""
        input_cost = self.usage["input"] * 0.001 * 0.005  # $0.005 per 1K input tokens
        output_cost = self.usage["output"] * 0.001 * 0.015  # $0.015 per 1K output tokens
        return input_cost + output_cost
    
    def get_stats(self):
        """Get current token usage statistics"""
        return {
            "input_tokens": self.usage["input"],
            "output_tokens": self.usage["output"],
            "estimated_cost": f"${self.get_cost_estimate():.4f}"
        }
