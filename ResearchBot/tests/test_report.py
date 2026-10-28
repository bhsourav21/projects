from researchbot.report import cited_numbers, executive_summary, slugify, sources_section

REPORT = """# Solar Power

## Executive Summary
Solar is cheap now [1]. Storage is improving [2, 3].

## Details
More text [1][3] and memory [M1].
"""

NOTES = [
    {"url": "https://a.com", "title": "A"},
    {"url": "https://b.com", "title": "B"},
    {"url": "https://c.com", "title": ""},
    {"url": "https://unused.com", "title": "Unused"},
]


def test_cited_numbers_handles_groups_and_order():
    assert cited_numbers(REPORT) == [1, 2, 3]
    assert cited_numbers(REPORT, prefix="M") == [1]


def test_sources_section_lists_only_cited_sources():
    hits = [{"question": "old q", "date": "2026-01-01T00:00:00", "report_path": "r.md"}]
    section = sources_section(REPORT, NOTES, hits)
    assert "https://a.com" in section and "https://c.com" in section
    assert "unused.com" not in section
    assert "[M1] Past research: old q (2026-01-01)" in section


def test_sources_section_ignores_out_of_range_citations():
    assert sources_section("Claim [9].", NOTES[:1], []) == ""


def test_executive_summary():
    assert executive_summary(REPORT) == "Solar is cheap now [1]. Storage is improving [2, 3]."


def test_executive_summary_fallback():
    assert executive_summary("# Title\nJust text.") == "Just text."


def test_slugify():
    assert slugify("What is RAG? (2026 edition)") == "what-is-rag-2026-edition"
    assert slugify("???") == "report"
