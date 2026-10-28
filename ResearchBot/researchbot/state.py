from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # short-term memory
    question: str  # current question, as the user typed it
    standalone_question: str  # question rewritten to make sense without the chat history
    memory_hits: list[dict]  # matches from Pinecone (research + conversation)
    strong_memory_hit: bool  # past research already covers the question well
    plan: dict  # {"queries": [...], "outline": [...], "rationale": str}
    plan_feedback: str  # the user's reason for rejecting / editing the plan
    plan_approved: bool
    search_results: list[dict]  # Tavily results (url, title, content, raw_content)
    search_retries: int
    page_notes: list[dict]  # condensed notes per page: {"url", "title", "notes"}
    report: str
    report_path: str
