from smartassist.tools.declaration import TOOL_SCHEMAS
from smartassist.tools.functions import calculate, get_current_date, web_search

# Maps tool name strings to their Python implementations for dispatch in agent.py
TOOL_MAP = {
    "calculate": lambda args: calculate(args["expression"]),
    "get_current_date": lambda _: get_current_date(),
    "web_search": lambda args: web_search(args["query"]),
}

__all__ = ["TOOL_SCHEMAS", "TOOL_MAP"]
