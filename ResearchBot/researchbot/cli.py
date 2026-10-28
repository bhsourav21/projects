"""Terminal chat loop: streams each step and asks for plan approval."""

import argparse
import sys
import uuid

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from researchbot.graph import build_graph, turn_input

console = Console()


def show_plan(payload: dict) -> None:
    plan = payload["plan"]
    queries = "\n".join(f"  {i}. {q}" for i, q in enumerate(plan["queries"], 1))
    outline = "\n".join(f"  - {h}" for h in plan["outline"])
    console.print(
        Panel(
            f"[bold]Question:[/] {payload['question']}\n\n"
            f"[bold]Search queries:[/]\n{queries}\n\n"
            f"[bold]Report outline:[/]\n{outline}\n\n"
            f"[bold]Why:[/] {plan['rationale']}",
            title="Research plan",
            border_style="cyan",
        )
    )


def ask_approval() -> dict:
    answer = console.input("[cyan]Approve plan? \\[y]es / or type what to change: [/]").strip()
    if answer.lower() in ("", "y", "yes"):
        return {"approved": True}
    return {"approved": False, "feedback": answer}


def show_update(node: str, update: dict) -> None:
    update = update or {}
    step = f"[bold green]▸ {node}[/]"
    if node == "memory_manage":
        removed = len(update.get("messages", []))
        if removed:
            console.print(f"{step} summarised {removed} older messages into long-term memory")
    elif node == "memory_retrieve":
        hits = update.get("memory_hits", [])
        console.print(f"{step} standalone question: {update.get('standalone_question')}")
        detail = ", ".join(f"{h['namespace']} {h['score']:.2f}" for h in hits) or "none"
        strong = ""
        if update.get("strong_memory_hit"):
            strong = " [yellow](strong match: fewer searches)[/]"
        console.print(f"  memory hits: {detail}{strong}")
    elif node == "planner":
        console.print(f"{step} drafted {len(update['plan']['queries'])} search queries")
    elif node == "human_approval":
        ok = update.get("plan_approved")
        console.print(f"{step} {'approved' if ok else 'rejected, replanning'}")
    elif node == "search":
        results = update.get("search_results", [])
        console.print(f"{step} found {len(results)} pages")
        for r in results:
            console.print(f"  - {r['url']}", style="dim")
    elif node == "reformulate":
        queries = "; ".join(update["plan"]["queries"])
        console.print(f"{step} no results, retry {update['search_retries']}: {queries}")
    elif node == "read":
        console.print(f"{step} {len(update.get('page_notes', []))} pages had relevant content")
    elif node == "synthesise":
        console.print(f"\n{step} report saved to {update['report_path']}")
    elif node == "save_to_pinecone":
        console.print(f"{step} done")
    elif node == "respond":
        console.print(Panel(update["messages"][-1].content, title="Answer", border_style="green"))


def run_turn(graph, question: str, config: dict, auto_approve: bool = False) -> dict:
    """Run one question to completion, handling the approval interrupt."""
    payload = turn_input(question)
    while True:
        pending = None
        streaming_report = False
        for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "messages"]):
            if mode == "messages":
                token, meta = chunk
                if meta.get("langgraph_node") == "synthesise" and token.content:
                    if not streaming_report:
                        console.print("[bold green]▸ synthesise[/] writing report...\n")
                        streaming_report = True
                    sys.stdout.write(token.content)
                    sys.stdout.flush()
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                else:
                    show_update(node, update)
        if pending is None:
            return graph.get_state(config).values
        show_plan(pending)
        decision = {"approved": True} if auto_approve else ask_approval()
        payload = Command(resume=decision)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchBot: autonomous research agent")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--thread", default=None, help="conversation id (default: new)")
    parser.add_argument("--yes", action="store_true", help="auto-approve research plans")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread or uuid.uuid4().hex[:8]}}

    if args.question:
        run_turn(graph, args.question, config, auto_approve=args.yes)
        return

    console.print(
        f"[bold]ResearchBot[/] (thread {config['configurable']['thread_id']}). "
        "Ask a research question, or type 'exit'."
    )
    while True:
        try:
            question = console.input("\n[bold magenta]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if question:
            run_turn(graph, question, config, auto_approve=args.yes)


if __name__ == "__main__":
    main()
