import re


def hybrid_chunk(
    text: str,
    sem_chunker,
    section_pattern: str = r"\n(?=#{1,6}\s)|\n\n+",
    min_chars: int = 100,
    max_chars: int = 2000,
) -> list[dict]:
    """
    Phase 1: regex structural split.
    Phase 2: semantic sub-split for oversized sections.
    Returns list of {'text', 'chars', 'method'} dicts.
    """
    final_chunks = []

    raw_sections = [s.strip() for s in re.split(section_pattern, text) if s.strip()]

    for section in raw_sections:
        if len(section) < min_chars:
            continue  # too short, skip
        elif len(section) <= max_chars:
            final_chunks.append({  # just right
                "text": section, "chars": len(section), "method": "regex"
            })
        else:
            # Phase 2: semantic sub-split
            sub = sem_chunker.split_text(section)
            for s in sub:
                if len(s.strip()) >= min_chars:
                    final_chunks.append({
                        "text": s.strip(),
                        "chars": len(s.strip()),
                        "method": "regex+semantic",
                    })

    return final_chunks
