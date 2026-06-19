# SmartAssist — Project Requirements

**PROJECT:** Build SmartAssist — AI assistant with tools, memory, and structured outputs

## Requirements

- Define at least 3 tools: a web search mock, a calculator (use Python eval safely), and `get_current_date()`. Register them with function calling.

- Build a multi-turn conversation loop: user inputs → agent decides to call a tool or respond → result fed back → next turn.

- For specific query types (e.g. 'create a task list'), return structured JSON using Pydantic. Render it nicely in the console.

- Add proper error handling: catch tool failures, LLM errors, and JSON parse errors — never let the app crash.

- Test with 10 diverse queries including ones that require tools and ones that don't. Fix failures.
