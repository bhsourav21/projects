"""End-to-end graph runs with the LLM, Tavily and Pinecone replaced by fakes."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from researchbot import graph as graph_module
from researchbot import nodes
from researchbot.config import settings


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Write a research report"):
            return AIMessage(
                "# Report\n\n## Executive Summary\nShort answer [1].\n\n## Body\nMore."
            )
        if prompt.startswith("These web search queries returned no results"):
            return AIMessage("broader query one\nbroader query two")
        return AIMessage("fake text")

    def batch(self, prompts, return_exceptions=False):
        return [AIMessage("- a relevant fact") for _ in prompts]

    def with_structured_output(self, schema):
        llm = self

        class Structured:
            def invoke(self, prompt):
                llm.prompts.append(prompt)
                return schema(queries=["q1", "q2"], outline=["Intro", "Findings"], rationale="r")

        return Structured()


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    llm = FakeLLM()
    calls = {"search": [], "summaries": [], "reports": []}
    search_results = {"empty_first": 0}

    def fake_search(query):
        calls["search"].append(query)
        if search_results["empty_first"] > 0:
            search_results["empty_first"] -= 1
            return []
        return [{"url": f"https://x.com/{query}", "title": query, "content": "c",
                 "raw_content": "page", "score": 1.0}]

    monkeypatch.setattr(nodes, "_llm", lambda: llm)
    monkeypatch.setattr(nodes, "_writer_llm", lambda: llm)
    monkeypatch.setattr(nodes, "web_search", fake_search)
    monkeypatch.setattr(nodes.memory, "search_memory", lambda q: [])
    monkeypatch.setattr(nodes.memory, "store_conversation_summary",
                        lambda s, t: calls["summaries"].append(s))
    monkeypatch.setattr(nodes.memory, "store_report",
                        lambda *a: calls["reports"].append(a) or 1)
    monkeypatch.setattr(settings, "reports_dir", tmp_path)
    return llm, calls, search_results


def run(g, question, config, decisions=None):
    decisions = list(decisions or [{"approved": True}])
    payload = graph_module.turn_input(question)
    while True:
        g.invoke(payload, config)
        state = g.get_state(config)
        if not state.interrupts:
            return state.values
        payload = Command(resume=decisions.pop(0))


def test_full_turn_produces_report_and_answer(fakes):
    _, calls, _ = fakes
    g = graph_module.build_graph()
    state = run(g, "What is RAG?", {"configurable": {"thread_id": "t1"}})

    assert state["report_path"].endswith(".md")
    assert "## Sources" in state["report"] and "https://x.com/q1" in state["report"]
    assert isinstance(state["messages"][-1], AIMessage)
    assert "Short answer" in state["messages"][-1].content
    assert len(calls["reports"]) == 1


def test_rejected_plan_goes_back_to_planner_with_feedback(fakes):
    llm, _, _ = fakes
    g = graph_module.build_graph()
    run(g, "What is RAG?", {"configurable": {"thread_id": "t2"}},
        decisions=[{"approved": False, "feedback": "focus on costs"}, {"approved": True}])

    planner_prompts = [p for p in llm.prompts if p.startswith("You are planning")]
    assert len(planner_prompts) == 2
    assert "focus on costs" in planner_prompts[1]


def test_empty_search_reformulates_then_continues(fakes):
    _, calls, search_results = fakes
    search_results["empty_first"] = 2  # first round (2 queries) returns nothing
    g = graph_module.build_graph()
    state = run(g, "obscure topic", {"configurable": {"thread_id": "t3"}})

    assert state["search_retries"] == 1
    assert "broader query one" in calls["search"]
    assert state["page_notes"]


def test_older_pairs_are_summarised_and_removed(fakes):
    _, calls, _ = fakes
    g = graph_module.build_graph()
    config = {"configurable": {"thread_id": "t4"}}
    for i in range(settings.short_term_pairs + 2):
        state = run(g, f"question {i}", config)

    humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    # 3 kept pairs + the question just answered (it was pending when trimming ran)
    assert humans == ["question 1", "question 2", "question 3", "question 4"]
    assert len(calls["summaries"]) == 1

    state = run(g, "question 5", config)
    humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["question 2", "question 3", "question 4", "question 5"]
    assert len(calls["summaries"]) == 2
