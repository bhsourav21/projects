"""LangGraph nodes and routing functions."""

from functools import lru_cache

from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from researchbot import memory, prompts
from researchbot.config import settings
from researchbot.report import executive_summary, save_report, sources_section
from researchbot.search import web_search
from researchbot.state import ResearchState


@lru_cache
def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, temperature=0)


@lru_cache
def _writer_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model, temperature=0.2, max_tokens=settings.max_report_tokens
    )


def _thread_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def _question(state: ResearchState) -> str:
    return state.get("standalone_question") or state["question"]


def _format_memory(hits: list[dict]) -> str:
    if not hits:
        return "(nothing relevant in memory)"
    return "\n\n".join(
        f"[M{i}] ({h['namespace']}, score {h['score']:.2f}) {h['text']}"
        for i, h in enumerate(hits, 1)
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def memory_manage(state: ResearchState, config: RunnableConfig) -> dict:
    """Keep the last N completed Q&A pairs; summarise older ones into Pinecone."""
    pairs, _pending = memory.split_pairs(state["messages"])
    if len(pairs) <= settings.short_term_pairs:
        return {}
    old = [m for pair in pairs[: -settings.short_term_pairs] for m in pair]
    summary = _llm().invoke(
        prompts.SUMMARISE_PROMPT.format(history=memory.format_history(old))
    ).content
    try:
        memory.store_conversation_summary(summary, _thread_id(config))
    except Exception as exc:
        print(f"[memory] could not store conversation summary: {exc}")
    return {"messages": [RemoveMessage(id=m.id) for m in old]}


def memory_retrieve(state: ResearchState) -> dict:
    """Make the question standalone, then search long-term memory for it."""
    history = memory.recent_pairs(state["messages"])
    question = state["question"]
    if history:
        question = _llm().invoke(
            prompts.STANDALONE_PROMPT.format(
                history=memory.format_history(history), question=question
            )
        ).content.strip()
    try:
        hits = memory.search_memory(question)
    except Exception as exc:
        print(f"[memory] search failed, continuing without memory: {exc}")
        hits = []
    strong = any(
        h["namespace"] == settings.research_namespace
        and h["score"] >= settings.strong_hit_threshold
        for h in hits
    )
    return {"standalone_question": question, "memory_hits": hits, "strong_memory_hit": strong}


class Plan(BaseModel):
    queries: list[str] = Field(description="Focused web search queries")
    outline: list[str] = Field(description="Section headings for the report")
    rationale: str = Field(description="Why this plan answers the question")


def planner(state: ResearchState) -> dict:
    feedback = state.get("plan_feedback", "")
    feedback_note = (
        prompts.FEEDBACK_NOTE.format(previous_plan=state.get("plan"), feedback=feedback)
        if feedback
        else ""
    )
    prompt = prompts.PLANNER_PROMPT.format(
        question=_question(state),
        history=memory.format_history(memory.recent_pairs(state["messages"])),
        memory=_format_memory(state.get("memory_hits", [])),
        coverage_note=prompts.STRONG_HIT_NOTE if state.get("strong_memory_hit") else "",
        feedback_note=feedback_note,
        max_queries=settings.max_queries,
    )
    plan: Plan = _llm().with_structured_output(Plan).invoke(prompt)
    plan.queries = plan.queries[: settings.max_queries]
    return {"plan": plan.model_dump()}


def human_approval(state: ResearchState) -> dict:
    """Pause and let the user approve or reject the plan.

    Resume with Command(resume={"approved": bool, "feedback": str}).
    """
    decision = interrupt({"question": _question(state), "plan": state["plan"]})
    if decision.get("approved"):
        return {"plan_approved": True, "plan_feedback": ""}
    return {"plan_approved": False, "plan_feedback": decision.get("feedback", "")}


def search(state: ResearchState) -> dict:
    results, seen = [], set()
    for query in state["plan"]["queries"]:
        for r in web_search(query):
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    return {"search_results": results[: settings.max_pages]}


def reformulate(state: ResearchState) -> dict:
    """Search found nothing: write broader queries and try again."""
    text = _llm().invoke(
        prompts.REFORMULATE_PROMPT.format(
            queries="\n".join(state["plan"]["queries"]),
            question=_question(state),
            n=min(3, settings.max_queries),
        )
    ).content
    queries = [q.strip("-• ").strip() for q in text.splitlines() if q.strip()]
    plan = {**state["plan"], "queries": queries[: settings.max_queries]}
    return {"plan": plan, "search_retries": state.get("search_retries", 0) + 1}


def read(state: ResearchState) -> dict:
    """Condense each page into notes relevant to the question."""
    results = state.get("search_results", [])
    if not results:
        return {"page_notes": []}
    prompt_inputs = [
        prompts.READ_PROMPT.format(
            question=_question(state),
            title=r["title"],
            content=(r["raw_content"] or r["content"])[: settings.max_page_chars],
        )
        for r in results
    ]
    replies = _llm().batch(prompt_inputs, return_exceptions=True)
    notes = []
    for r, reply in zip(results, replies):
        if isinstance(reply, Exception):
            print(f"[read] failed to read {r['url']}: {reply}")
            continue
        text = reply.content.strip()
        if text and "NOT RELEVANT" not in text[:20]:
            notes.append({"url": r["url"], "title": r["title"], "notes": text})
    return {"page_notes": notes}


def synthesise(state: ResearchState) -> dict:
    notes = state.get("page_notes", [])
    hits = state.get("memory_hits", [])
    sources = "\n\n".join(
        f"[{i}] {n['title']} ({n['url']})\n{n['notes']}" for i, n in enumerate(notes, 1)
    )
    prompt = prompts.SYNTHESISE_PROMPT.format(
        question=_question(state),
        history=memory.format_history(memory.recent_pairs(state["messages"])),
        outline="\n".join(f"- {h}" for h in state["plan"]["outline"]),
        sources=sources or "(no web sources found)",
        memory=_format_memory(hits),
    )
    report = _writer_llm().invoke(prompt).content.strip()
    report += sources_section(report, notes, hits)
    path = save_report(report, _question(state))
    return {"report": report, "report_path": str(path)}


def save_to_pinecone(state: ResearchState, config: RunnableConfig) -> dict:
    urls = [n["url"] for n in state.get("page_notes", [])]
    try:
        stored = memory.store_report(
            state["report"], _question(state), urls, state["report_path"], _thread_id(config)
        )
        if stored == 0:
            print("[memory] a near-identical report is already stored, skipped")
    except Exception as exc:
        print(f"[memory] could not store report: {exc}")
    return {}


def respond(state: ResearchState) -> dict:
    """Short answer for the chat; this is what short-term memory keeps."""
    summary = executive_summary(state["report"])
    return {"messages": [AIMessage(f"{summary}\n\nFull report: {state['report_path']}")]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_approval(state: ResearchState) -> str:
    return "search" if state.get("plan_approved") else "planner"


def route_after_search(state: ResearchState) -> str:
    if state.get("search_results"):
        return "read"
    if state.get("search_retries", 0) < settings.max_search_retries:
        return "reformulate"
    return "read"
