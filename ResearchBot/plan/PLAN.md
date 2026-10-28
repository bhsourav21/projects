# ResearchBot — Plan

## Goal

Given a research question, the agent works on its own to:

- search the web with Tavily
- read the pages it finds
- pull relevant past research from a vector store (Pinecone)
- write a structured Markdown report

Requirements:

- Built with LangGraph, with conditional edges and a human approval step before anything runs.
- Short-term memory (recent conversation in state) and long-term memory (past research and older conversation stored in Pinecone and searched by meaning).
- Each step's progress streams to the terminal. The final output is a well-formatted Markdown report.
- Test with 5 different research questions and fix the 2 hardest failure cases.

## Decisions

| Topic | Decision |
|---|---|
| Routing | Plain Python routing functions (`add_conditional_edges`), with no LLM supervisor |
| Critic agent | Left out for now |
| Session type | A chat session with follow-up questions, using the same `thread_id` across turns |
| Human approval | Approve the **plan**, before any search runs |
| Short-term memory | The last **3 question-and-answer exchanges** (6 messages) are passed to the LLM word for word |
| Long-term memory | Older exchanges are summarised and saved to Pinecone. Final reports are saved to Pinecone too. |

## Setup

- `.env` contains `OPENAI_API_KEY`, `TAVILY_API_KEY` and `PINECONE_API_KEY`.
- The Pinecone index dimension must match the embedding model (`text-embedding-3-small` uses 1536).
- Pinecone namespaces:
  - `conversation`: summaries of older exchanges
  - `research`: final reports and their sources

## State

```python
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]  # short-term memory
    question: str                            # current question
    memory_hits: list       # matches from Pinecone (research + conversation)
    plan: dict              # search queries + report outline
    plan_feedback: str      # the user's reason for rejecting or editing the plan
    search_results: list    # Tavily results (url, title, content)
    search_retries: int
    page_notes: list        # short notes from each page, with its source url
    report_path: str
```

A checkpointer (`MemorySaver`) with a `thread_id` keeps state between turns. `interrupt()` also needs it.

## Memory design

- **Short-term:** the last 3 exchanges (6 messages) stay word for word in `messages`.
- **Older conversation:** once there are more than 3 finished exchanges, the oldest is summarised by the LLM, saved to the `conversation` namespace, and removed from state with `RemoveMessage`. Count only finished exchanges, so the question currently being answered is never summarised.
- **Research:** each final report is split into chunks, embedded, and saved to the `research` namespace with metadata `{question, date, urls, thread_id}`. If a near-identical report already exists, it's skipped so the same question doesn't create duplicates.

## Nodes

| # | Node | What it does |
|---|---|---|
| 1 | `memory_manage` | If there are more than 3 finished exchanges, summarise the older ones, save them to `conversation`, and remove them from state |
| 2 | `memory_retrieve` | Find the top 5 matches across both namespaces, searching with the question plus the last 3 exchanges. This lets follow-ups like "what about its cost?" find the right material. |
| 3 | `planner` | The LLM writes the search queries and a report outline, using `memory_hits` and `plan_feedback` |
| 4 | `human_approval` | Show the plan with `interrupt()`. The user approves it or rejects it with feedback. |
| 5 | `search` | Run a Tavily search for each planned query |
| 6 | `read` | Get page content (Tavily raw content or extract) and boil each page down to notes relevant to the question, keeping the source url |
| 7 | `synthesise` | The LLM writes a Markdown report of at most 2000 tokens, with citations, from `page_notes`, `memory_hits` and the last 3 exchanges. Saved to `reports/<slug>_<timestamp>.md`. |
| 8 | `save_to_pinecone` | Split the report into chunks, embed them, and save them to `research`, skipping near-duplicates |
| 9 | `respond` | Send the user a short summary and the report path |

## Graph

```
START → memory_manage → memory_retrieve → planner → human_approval
                                             ▲            │
                                             └─(reject)───┤
                                                          │(approve)
                                                          ▼
                              ┌──(no results, retries < 2)─ search
                              └──────────────────────────▶   │(results ok / retries used up)
                                                             ▼
                          read → synthesise → save_to_pinecone → respond → END
```

Conditional edges:

- `human_approval`: rejected goes to `planner` (with feedback). Approved goes to `search`.
- `search`: no results with retries left goes back to `search` with a reworded query. Results found, or retries used up, goes to `read`.
- Optional: when `memory_retrieve` finds a strong match (score above a threshold), the planner is told to plan fewer, narrower searches.

## Streaming

- `graph.stream(..., stream_mode="updates")` prints each node's progress.
- `stream_mode="messages"` streams the final report token by token.

## Testing

Five different questions:

1. A factual question about recent news
2. A comparison (A vs B)
3. A technical how-to
4. A vague or ambiguous question
5. A follow-up that depends on an earlier turn

Failure cases to watch for:

- Search returns nothing
- Pages so long they overflow the context window
- The first run, when Pinecone is still empty
- A follow-up that loses track of what it refers to (especially after older exchanges are summarised)
- A report longer than 2000 tokens

Pick the 2 hardest failures, fix them, and write down the behaviour before and after each fix.

## Out of scope (for now)

- A critic or reviewer agent that sends weak reports back for more research
