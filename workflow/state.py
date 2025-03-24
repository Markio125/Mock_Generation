from typing import TypedDict, Annotated, List, Dict

class GraphState(TypedDict):
    total_questions: Annotated[int, "total questions needed"]
    distribution: Annotated[Dict[str, int], "questions per topic"]
    context: Annotated[Dict[str, List[str]], "retrieved context per topic"]
    questions: Annotated[Dict[str, List[str]], "generated questions"]
    remaining_topics: Annotated[List[str], "topics left to process"]
    detected_topics: Annotated[List[str], "automatically detected topics"]
