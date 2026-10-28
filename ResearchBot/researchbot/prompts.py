STANDALONE_PROMPT = """Rewrite the user's latest question so it can be understood without \
the conversation history. Replace pronouns and references ("it", "that", "the second one") \
with what they refer to. If the question already stands on its own, return it unchanged.
Return only the rewritten question.

Conversation:
{history}

Latest question: {question}"""

SUMMARISE_PROMPT = """Summarise this earlier part of a research conversation in 3-6 sentences. \
Keep the questions asked, the key findings and any report paths, so it can be recalled later.

{history}"""

PLANNER_PROMPT = """You are planning a web research task.

Question: {question}

Recent conversation:
{history}

Relevant past research and conversation (from long-term memory):
{memory}

{coverage_note}
{feedback_note}
Produce:
- queries: 1-{max_queries} focused web search queries that together answer the question.
  Don't repeat what memory already covers well.
- outline: 3-6 section headings for the final report.
- rationale: one or two sentences on why this plan."""

STRONG_HIT_NOTE = """Note: past research already covers this question well. \
Plan only 1-2 narrow queries to fill gaps or get newer information."""

FEEDBACK_NOTE = """The user rejected the previous plan.
Previous plan: {previous_plan}
User feedback: {feedback}
Revise the plan to address the feedback."""

REFORMULATE_PROMPT = """These web search queries returned no results:
{queries}

Original question: {question}

Write {n} different, broader or simpler search queries likely to return results. \
One per line, no numbering, nothing else."""

READ_PROMPT = """Extract the information from this web page that helps answer the question.
Write concise bullet-point notes (at most ~250 words). Keep concrete facts, numbers, dates \
and names. If the page has nothing relevant, reply exactly: NOT RELEVANT

Question: {question}

Page title: {title}
Page content:
{content}"""

SYNTHESISE_PROMPT = """Write a research report in Markdown answering the question.

Question: {question}

Recent conversation (for context on follow-up questions):
{history}

Suggested outline:
{outline}

Web sources (cite as [1], [2] ...):
{sources}

Past research from memory (cite as [M1], [M2] ...):
{memory}

Rules:
- Start with "# <title>", then a "## Executive Summary" section of 2-4 sentences.
- Then follow the outline with "##" sections. Use bullet points and tables where they help.
- Every factual claim must cite a source like [1] or [M1]. Don't invent facts or sources.
- If the sources don't answer part of the question, say so plainly in a
  "## Gaps and Limitations" section.
- Stay under 1200 words. Do NOT write a sources/references list; it is added automatically."""
