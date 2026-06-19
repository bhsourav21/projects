# SmartAssist

A minimal AI assistant built with the raw OpenAI SDK that demonstrates tool calling, structured output, and multi-turn conversation history — all wired up manually without any agent framework.

---

## What it does

SmartAssist runs a fixed sequence of 10 user prompts through a single shared conversation thread. For each prompt the agent:

1. Sends the message history to `gpt-4o-mini`
2. If the model requests a tool (calculator, date lookup, web search), executes it and sends results back in a follow-up call
3. Parses the final response into a typed `AgentResponse` (either plain text or a numbered task list)
4. Renders it to the console

---

## Architecture

```
main.py
  │
  │  for each prompt in user_prompts:
  │    append HumanMessage → call run_turn()
  │
  └─► agent.py: run_turn()
        │
        │  [1] First API call
        │      model = gpt-4o-mini
        │      tools = [calculate, get_current_date, web_search]
        │      response_format = AgentResponse
        │
        ├── tool_calls present?
        │     │
        │    YES
        │     │
        │     ├─► agent.py: handle_tool_calls()
        │     │     │
        │     │     └─► tools/functions.py
        │     │           calculate(expression)      → eval arithmetic
        │     │           get_current_date()         → date.today()
        │     │           web_search(query)          → mock keyword lookup
        │     │
        │     └─► [2] Follow-up API call (no tools, structured output only)
        │               → parsed AgentResponse
        │
        └── NO tool_calls
              └─► parsed AgentResponse from first call directly
                    │
                    └─► renderer.py: render_response()
                          response_type == "text"      → print text
                          response_type == "task_list" → print numbered table
```

### File map

```
SmartAssist/
├── main.py                        entry point — conversation loop
└── smartassist/
    ├── config.py                  OpenAI client, model name, max tokens
    ├── models.py                  AgentResponse + Task Pydantic models
    ├── prompts.py                 system prompt + 10 user prompts
    ├── agent.py                   run_turn(), handle_tool_calls()
    ├── renderer.py                render_response(), print_task_list()
    └── tools/
        ├── declaration.py         TOOL_SCHEMAS (OpenAI JSON schema format)
        ├── functions.py           tool implementations
        └── __init__.py            TOOL_SCHEMAS + TOOL_MAP exports
```

---

## Prerequisites

- Python 3.10+
- An OpenAI API key

Install dependencies:

```bash
cd SmartAssist
pip install -e .
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## How to run

```bash
uv venv .venv
uv sync
uv run python main.py
```

The script runs all 10 prompts sequentially and prints each response.

---

## Example queries and expected output

### 1. Tool use — date lookup

**Prompt:** `What is the date today?`

```
--------------------------------------------
Turn 1
User     : What is the date today?
Today's date is June 18, 2026.
```

Flow: model calls `get_current_date()` → gets `"2026-06-18"` → follow-up call produces text response.

---

### 2. Structured output — task list

**Prompt:** `Prepare a task list for visiting a Gynaecologist`

```
--------------------------------------------
Turn 3
User     : Prepare a task list for visiting a Gynaecologist
  #     Task
  ----- --------------------------------------------------
  1     Schedule an appointment with a gynaecologist
  2     Gather your medical history and previous records
  3     List any symptoms or concerns to discuss
  4     Arrange transportation to the clinic
  5     Bring your insurance card and ID
  6     Note any medications you are currently taking
  7     Follow any pre-visit instructions given by the clinic
  8     Attend the appointment on time
```

Flow: no tool called — model sets `response_type = "task_list"` and populates `tasks` directly from first call.

---

### 3. No tool — general knowledge

**Prompt:** `What is the difference between a virus and a bacterium? Keep it simple.`

```
--------------------------------------------
Turn 5
User     : What is the difference between a virus and a bacterium? Keep it simple.
Bacteria are single-celled living organisms that can reproduce on their own and can
be killed by antibiotics. Viruses are not living cells — they are tiny packets of
genetic material that need to hijack a host cell to replicate, and antibiotics have
no effect on them.
```

Flow: no tool called — model returns a plain text `AgentResponse` directly from the first call.

---

## What I learned

**Tool calling is a two-call pattern, not one.**
When the model wants to use a tool, it doesn't answer — it returns a list of tool calls and stops. We need to execute the tools, append the results to the message history as `tool` role messages (each linked back to the model's request via `tool_call_id`), and after that make a second API call to get the actual answer. Understanding this made the whole agentic loop click for me.

**The model decides when to use a tool — we just offer them.**
We pass the tool schemas upfront and the model chooses whether to call any of them based on the query. The same prompt sent twice might or might not trigger a tool call depending on how the model interprets it. This is different from deterministic function dispatch — the routing is probabilistic.

**Structured output and tool calling conflict if we are not careful.**
I initially tried to enforce `response_format=AgentResponse` on the first call alongside tools. That works when the model answers directly, but if the model calls a tool, the structured format is ignored. The clean solution is two separate model calls: the first one offers tools and lets the model respond freely, the second one (after tool results are injected) enforces the structured schema.

**Message history is the entire memory of the agent.**
There's no hidden state — everything the model "knows" about the conversation is in the list we pass on each call. Adding a message appends context; forgetting to append a message means it's as if that turn never happened. This made me think carefully about what to append and when, especially the assistant's tool-call message (which must precede the tool result messages).

**Pydantic as a contract between the agent and the model.**
Using `AgentResponse` with a `response_type` discriminator field meant I never had to guess what format the model returned — it was enforced at parse time. The model had to signal its intent (`"text"` vs `"task_list"`) explicitly, which made the renderer simple and reliable.
