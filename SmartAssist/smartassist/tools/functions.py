from datetime import date
import re

# Mock results keyed by keyword — simulates a real search API without network calls
MOCK_SEARCH_DB = {
    "python": "Python 3.13 was released in October 2024 with improved error messages and a new JIT compiler.",
    "openai": "OpenAI offers GPT-4o, GPT-4o-mini, and o1 models via its API as of 2024.",
    "gynaecologist": "A gynaecologist specialises in female reproductive health, covering routine check-ups, screenings, and treatment.",
}


def calculate(expression: str) -> float:
    """Evaluates an arithmetic expression string safely — only digits, operators, and parentheses are allowed."""
    if not re.fullmatch(r'[\d\s\+\-\*\/\.\(\)\%]+', expression):
        raise ValueError(f"Unsafe expression rejected: '{expression}'")
    return eval(expression)


def get_current_date() -> str:
    """Returns today's date as a YYYY-MM-DD string."""
    return date.today().isoformat()


def web_search(query: str) -> str:
    """Returns a mock search result for the query by matching keywords against a static database."""
    query_lower = query.lower()
    for keyword, result in MOCK_SEARCH_DB.items():
        if keyword in query_lower:
            return result
    return f"No results found for '{query}'. (mock search)"
