from unittest.mock import MagicMock

from api.services.chunking import hybrid_chunk


def test_hybrid_chunk_short_section_is_dropped():
    sem_chunker = MagicMock()
    chunks = hybrid_chunk("short", sem_chunker=sem_chunker, min_chars=10, max_chars=100)
    assert chunks == []
    sem_chunker.split_text.assert_not_called()


def test_hybrid_chunk_regular_section_uses_regex_only():
    text = "A" * 150
    sem_chunker = MagicMock()
    chunks = hybrid_chunk(text, sem_chunker=sem_chunker, min_chars=10, max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0]["method"] == "regex"
    sem_chunker.split_text.assert_not_called()


def test_hybrid_chunk_oversized_section_uses_semantic_chunker():
    text = "B" * 3000
    sem_chunker = MagicMock()
    sem_chunker.split_text.return_value = ["B" * 1500, "B" * 1500]
    chunks = hybrid_chunk(text, sem_chunker=sem_chunker, min_chars=10, max_chars=2000)
    assert len(chunks) == 2
    assert all(c["method"] == "regex+semantic" for c in chunks)
    sem_chunker.split_text.assert_called_once_with(text)
