"""Run the 5 test questions in one conversation and record metrics.

Usage: uv run python -m eval.run_eval
Plans are auto-approved. Results go to eval/results.md.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path

import tiktoken

from researchbot.cli import run_turn
from researchbot.config import settings
from researchbot.graph import build_graph
from researchbot.report import cited_numbers

# One conversation: Q3 is a follow-up to Q2, and Q5 triggers short-term memory trimming.
QUESTIONS = [
    ("recent / factual", "What are the latest developments in solid-state batteries for EVs?"),
    ("comparison", "Compare LangGraph, CrewAI and AutoGen for building multi-agent systems."),
    ("follow-up", "Which of them is easiest for a beginner, and why?"),
    ("technical how-to", "How do I set up hybrid search (dense + sparse) in Pinecone?"),
    ("vague", "What's the best database?"),
]

RESULTS = Path(__file__).parent / "results.md"


def main() -> None:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    graph = build_graph()
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:6]}"}}
    rows, details = [], []

    for kind, question in QUESTIONS:
        print(f"\n{'=' * 80}\n[{kind}] {question}\n{'=' * 80}")
        start = time.time()
        try:
            state = run_turn(graph, question, config, auto_approve=True)
        except Exception as exc:
            rows.append(f"| {kind} | ERROR: {exc} | | | | | | |")
            continue
        report = state.get("report", "")
        body = report.split("\n## Sources")[0]
        tokens = len(enc.encode(body))
        issues = []
        if tokens > settings.max_report_tokens:
            issues.append("over token limit")
        if not cited_numbers(report) and not cited_numbers(report, prefix="M"):
            issues.append("no citations")
        if not state.get("page_notes"):
            issues.append("no web sources")
        rows.append(
            f"| {kind} | {len(state['plan']['queries'])} | {len(state.get('search_results', []))} "
            f"| {len(state.get('page_notes', []))} | {state.get('search_retries', 0)} "
            f"| {tokens} | {time.time() - start:.0f}s | {', '.join(issues) or 'ok'} |"
        )
        hits = [(h["namespace"], round(h["score"], 2)) for h in state.get("memory_hits", [])]
        details.append(
            f"### [{kind}] {question}\n\n"
            f"- standalone question: {state.get('standalone_question')}\n"
            f"- memory hits: {hits}\n"
            f"- queries: {state['plan']['queries']}\n"
            f"- report: `{state.get('report_path')}`\n"
        )

    RESULTS.write_text(
        f"# Eval run {datetime.now():%Y-%m-%d %H:%M}\n\n"
        "| type | queries | pages | relevant | retries | report tokens | time | issues |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n" + "\n".join(details),
        encoding="utf-8",
    )
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
