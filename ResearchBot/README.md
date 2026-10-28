# ResearchBot

An autonomous research agent built with LangGraph. Give it a research question and it:

- plans its searches and waits for your approval
- searches the web with Tavily and reads the pages it finds
- pulls related past research from Pinecone
- writes a cited Markdown report

It's a chat: follow-up questions work, and older conversation is summarised into long-term memory.

The design is described in [plan/PLAN.md](plan/PLAN.md).

## Graph

```
START → memory_manage → memory_retrieve → planner → human_approval ─(reject)→ planner
                                                         │(approve)
                                                         ▼
                                   reformulate ←(no results, retries left)─ search
                                        └──────────────→ search ─(results / retries used up)→ read
read → synthesise → save_to_pinecone → respond → END
```

| Node | What it does |
|---|---|
| `memory_manage` | Keeps the last 3 Q&A pairs word for word. Older pairs are summarised into Pinecone (`conversation` namespace) and removed from state. |
| `memory_retrieve` | Rewrites follow-ups as standalone questions, then searches both Pinecone namespaces |
| `planner` | Writes search queries and a report outline. Plans fewer searches when memory already covers the question. |
| `human_approval` | `interrupt()`: you approve the plan, or type what to change |
| `search` / `reformulate` | Tavily search. Retries with broader queries if nothing comes back. |
| `read` | Boils each page down to notes relevant to the question |
| `synthesise` | Markdown report (≤ 2000 tokens) with `[n]` citations, saved to `reports/` |
| `save_to_pinecone` | Chunks and embeds the report into the `research` namespace, skipping near-duplicates |
| `respond` | Chat answer: the executive summary plus the report path |

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # add OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY
```

The Pinecone index (`researchbot-index`, 1536 dimensions, cosine) is created automatically on first use.

## Usage

```bash
uv run python main.py                       # interactive chat
uv run python main.py "What is GraphRAG?"   # one question
uv run python main.py --yes                 # auto-approve plans
uv run python main.py --thread mythread     # name the conversation thread
```

Each step streams to the terminal as it runs, and the report streams token by token.

Short-term memory lives in an in-memory checkpointer, so it lasts for one run of the program. Long-term memory (Pinecone) stays between runs.

## Tests and evaluation

```bash
uv run pytest                    # unit + graph-flow tests (no API calls)
uv run python -m eval.run_eval   # the 5 test questions, real APIs → eval/results.md
```

The eval asks 5 different kinds of question in one conversation:

1. recent/factual
2. comparison
3. a follow-up to question 2
4. technical how-to
5. vague

It records queries, pages, retries, report token count and time for each, and flags these issues: over the token limit, no citations, no web sources.

## Settings

All settings are in [researchbot/config.py](researchbot/config.py). Each one can be overridden with an environment variable of the same name, e.g. `CHAT_MODEL=gpt-4.1-mini`.
