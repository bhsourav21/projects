# Tool declarations passed to the OpenAI API — defines what tools the model can call and their argument schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression. Supports +, -, *, /",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate, e.g. '10 + 2 - 5'",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Return current date in YYYY-MM-DD format",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information on a topic and return a summary",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'latest Python version'",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
