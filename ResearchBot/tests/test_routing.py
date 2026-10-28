from researchbot.config import settings
from researchbot.graph import build_graph
from researchbot.nodes import route_after_approval, route_after_search


def test_route_after_approval():
    assert route_after_approval({"plan_approved": True}) == "search"
    assert route_after_approval({"plan_approved": False}) == "planner"


def test_route_after_search():
    assert route_after_search({"search_results": [{"url": "x"}]}) == "read"
    assert route_after_search({"search_results": [], "search_retries": 0}) == "reformulate"
    exhausted = {"search_results": [], "search_retries": settings.max_search_retries}
    assert route_after_search(exhausted) == "read"


def test_graph_compiles_with_expected_nodes():
    nodes = set(build_graph().get_graph().nodes)
    expected = {
        "memory_manage", "memory_retrieve", "planner", "human_approval", "search",
        "reformulate", "read", "synthesise", "save_to_pinecone", "respond",
    }
    assert expected <= nodes
