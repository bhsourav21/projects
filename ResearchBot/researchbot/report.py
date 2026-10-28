"""Report formatting helpers (pure functions, no API calls)."""

import re
from datetime import datetime
from pathlib import Path

from researchbot.config import settings


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "report"


def cited_numbers(report: str, prefix: str = "") -> list[int]:
    """Citation numbers used in the report, in first-seen order.

    Handles [1], [1, 3] and [1][2]. With prefix="M" it reads memory citations like [M2].
    """
    seen: list[int] = []
    for group in re.findall(rf"\[({prefix}\d+(?:\s*,\s*{prefix}\d+)*)\]", report):
        for part in group.split(","):
            n = int(part.strip().removeprefix(prefix))
            if n not in seen:
                seen.append(n)
    return seen


def sources_section(report: str, notes: list[dict], memory_hits: list[dict]) -> str:
    """A Markdown sources list containing only the sources the report actually cites."""
    lines = []
    for n in cited_numbers(report):
        if 1 <= n <= len(notes):
            note = notes[n - 1]
            lines.append(f"[{n}] [{note['title'] or note['url']}]({note['url']})")
    for n in cited_numbers(report, prefix="M"):
        if 1 <= n <= len(memory_hits):
            hit = memory_hits[n - 1]
            label = hit.get("question") or "earlier conversation"
            where = f" — `{hit['report_path']}`" if hit.get("report_path") else ""
            lines.append(f"[M{n}] Past research: {label} ({hit.get('date', '')[:10]}){where}")
    if not lines:
        return ""
    return "\n\n## Sources\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def executive_summary(report: str) -> str:
    """The text of the "## Executive Summary" section, or the report's opening as a fallback."""
    match = re.search(r"##\s*Executive Summary\s*\n(.*?)(?=\n##\s|\Z)", report, re.S | re.I)
    if match:
        return match.group(1).strip()
    body = re.sub(r"^#.*\n", "", report.strip()).strip()
    return body[:600]


def save_report(report: str, question: str) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = settings.reports_dir / f"{slugify(question)}_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    return path
