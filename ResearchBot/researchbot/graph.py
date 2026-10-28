from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from researchbot import nodes
from researchbot.state import ResearchState


def build_graph(checkpointer=None):
    g = StateGraph(ResearchState)

    g.add_node("memory_manage", nodes.memory_manage)
    g.add_node("memory_retrieve", nodes.memory_retrieve)
    g.add_node("planner", nodes.planner)
    g.add_node("human_approval", nodes.human_approval)
    g.add_node("search", nodes.search)
    g.add_node("reformulate", nodes.reformulate)
    g.add_node("read", nodes.read)
    g.add_node("synthesise", nodes.synthesise)
    g.add_node("save_to_pinecone", nodes.save_to_pinecone)
    g.add_node("respond", nodes.respond)

    g.add_edge(START, "memory_manage")
    g.add_edge("memory_manage", "memory_retrieve")
    g.add_edge("memory_retrieve", "planner")
    g.add_edge("planner", "human_approval")
    g.add_conditional_edges(
        "human_approval", nodes.route_after_approval, ["search", "planner"]
    )
    g.add_conditional_edges("search", nodes.route_after_search, ["read", "reformulate"])
    g.add_edge("reformulate", "search")
    g.add_edge("read", "synthesise")
    g.add_edge("synthesise", "save_to_pinecone")
    g.add_edge("save_to_pinecone", "respond")
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())


def turn_input(question: str) -> dict:
    """State for a new question; resets the per-turn fields from the previous turn."""
    return {
        "messages": [HumanMessage(question)],
        "question": question,
        "standalone_question": "",
        "plan_feedback": "",
        "plan_approved": False,
        "search_results": [],
        "search_retries": 0,
        "page_notes": [],
    }
